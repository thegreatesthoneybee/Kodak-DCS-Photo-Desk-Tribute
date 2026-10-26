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
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    if head[:4] not in (b"MM\x00*", b"II*\x00"):
        return "", ""
    bo = ">" if head[:2] == b"MM" else "<"
    off, = struct.unpack_from(bo + "I", head, 4)
    if off + 2 > len(head):
        return "", ""
    n, = struct.unpack_from(bo + "H", head, off)
    vals = {}
    for i in range(n):
        p = off + 2 + 12 * i
        if p + 12 > len(head):
            break
        tag, typ, cnt, val = struct.unpack_from(bo + "HHII", head, p)
        if tag in (0x10F, 0x110) and typ == 2:
            raw = head[val:val + cnt] if cnt > 4 else head[p + 8:p + 8 + cnt]
            vals[tag] = raw.split(b"\x00")[0].decode("latin1").strip()
    return vals.get(0x10F, ""), vals.get(0x110, "")


def source_profile(model, illuminant):
    if illuminant not in ILLUMINANTS:
        raise ValueError(f"unknown illuminant {illuminant}")
    for key, pattern in _FAMILIES:
        if key.lower() in model.lower():
            return pattern.format(illuminant)
    return None
