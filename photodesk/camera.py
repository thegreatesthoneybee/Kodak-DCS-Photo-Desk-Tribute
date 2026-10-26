"""Mapping from DCS camera models to Kodak source-correction profile families."""
import struct

ILLUMINANTS = ("Daylight", "Flash", "Tungsten", "Fluorescent")

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

_TONE_CURVES = (
    ("SLR/c", "DCSProSLRcToneCurve.icm"),
    ("SLR/n", "DCSProSLRnToneCurve.icm"),
    ("Pro 14", "DCSPro14ToneCurve.icm"),
)


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


def tone_curve_profile(model):
    for key, name in _TONE_CURVES:
        if key.lower() in model.lower():
            return name
    return None
