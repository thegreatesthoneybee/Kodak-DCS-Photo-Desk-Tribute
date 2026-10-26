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


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """Processing options Photo Desk offers for a camera model."""
    dcs: bool              # DCS raw file (all processing options); False for other cameras
    pro_back: bool         # DCS Pro Back: +-1 EV, no Custom balance, no Auto Scene Balance
    pro_slr: bool          # DCS Pro 14n/14nx/SLR: Advanced/Moire/Expert noise reduction
    lens_optimization: bool

    @property
    def exposure_range(self):
        return 1.0 if self.pro_back else 2.0

    @property
    def noise_types(self):
        if self.pro_slr:
            return ("none", "advanced", "moire", "expert")
        return ("none", "basic", "advanced", "moire")

    @property
    def balances(self):
        return ("preset", "click", "auto") if self.pro_back else ("preset", "click", "custom", "auto")


def capabilities(model):
    m = (model or "").lower()
    pro_back = "proback" in m or "pro back" in m
    pro_slr = any(k in m for k in ("slr/", "pro 14", "pro14"))
    dcs = pro_back or pro_slr or "dcs" in m
    lens = any(k in m for k in ("14nx", "slr/n", "slr/c"))
    return Capabilities(dcs, pro_back, pro_slr, lens)


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
