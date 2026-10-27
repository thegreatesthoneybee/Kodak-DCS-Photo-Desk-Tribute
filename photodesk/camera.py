"""DCS camera models: TIFF Make/Model reading, source-correction profiles and capabilities."""
import dataclasses
import struct

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


def _ifd0_strings(path, tags):
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    if head[:4] not in (b"MM\x00*", b"II*\x00"):
        return {}
    bo = ">" if head[:2] == b"MM" else "<"
    off, = struct.unpack_from(bo + "I", head, 4)
    if off + 2 > len(head):
        return {}
    n, = struct.unpack_from(bo + "H", head, off)
    vals = {}
    for i in range(n):
        p = off + 2 + 12 * i
        if p + 12 > len(head):
            break
        tag, typ, cnt, val = struct.unpack_from(bo + "HHII", head, p)
        if tag in tags and typ == 2:
            raw = head[val:val + cnt] if cnt > 4 else head[p + 8:p + 8 + cnt]
            vals[tag] = raw.split(b"\x00")[0].decode("latin1").strip()
    return vals


_TIFF_INTS = {1: "B", 3: "H", 4: "I", 8: "h", 9: "i", 5: "II", 10: "ii"}   # 5/10: rationals
KODAK_IFD = 0x8290
# Kodak IFD tags 2654-2656: [56, 0, strength, 0] per channel, the Lens Optimization strength the
# camera applied (DCSPro3SLR/DCSPro4SLR.dll read them and accept strengths of 0..99).
LENS_STRENGTH_TAG = 2654
# Kodak IFD tags 2120-2125: camera RGB of a neutral under each white balance setting of the camera
# (Daylight, Tungsten, Fluorescent, Flash, Custom, Auto); tag 1020 is the setting used for the shot.
# Photo Desk takes the Preset balance from the lighting's point and Custom from tag 2124 (0x84C).
BALANCE_POINT_TAGS = {"Daylight": 2120, "Tungsten": 2121, "Fluorescent": 2122, "Flash": 2123,
                      "custom": 2124, "camera_auto": 2125}
WB_SETTING_TAG = 1020
WB_SETTINGS = ("Daylight", "Tungsten", "Fluorescent", "Flash", "custom", "camera_auto")


def read_kodak_tags(path, tags):
    """Integer values of the given Kodak IFD (tag 0x8290) tags as {tag: tuple}; reads only the
    IFD entries and those values, not the whole file."""
    out = {}
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:4] not in (b"MM\x00*", b"II*\x00"):
            return out
        bo = ">" if head[:2] == b"MM" else "<"

        def entries(off):
            f.seek(off)
            raw = f.read(2)
            if len(raw) < 2:
                return []
            n, = struct.unpack(bo + "H", raw)
            data = f.read(12 * n)
            return [struct.unpack_from(bo + "HHI4s", data, 12 * i) for i in range(len(data) // 12)]

        kodak = next((v for t, typ, _, v in entries(struct.unpack(bo + "I", head[4:])[0])
                      if t == KODAK_IFD and typ in (4, 13)), None)
        if kodak is None:
            return out
        for tag, typ, cnt, val in entries(struct.unpack(bo + "I", kodak)[0]):
            code = _TIFF_INTS.get(typ)
            if tag not in tags or code is None or cnt > 64:
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
                    v = tuple(n / d if d else 0.0 for n, d in zip(v[::2], v[1::2]))
                out[tag] = v
    return out


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
        v = read_kodak_tags(path, (LENS_STRENGTH_TAG,)).get(LENS_STRENGTH_TAG)
        if v and len(v) >= 3 and 0 <= v[2] <= 99:
            info["lens_optimization"] = v[2]
    return info


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
