"""DCS white balance (.wb) files, as written by Photo Desk's Image > Save Balance File.

The layout follows the writer in the original PhotoDsk.exe (KPDC SDK routine called from
``DCSImage::WriteWhiteBalanceFile``): a big-endian TIFF whose IFD0 carries eight identification
tags of the source raw file and a Kodak IFD (tag 0x8290) with 37 colour tags of that file, where
tag 0x40c holds the click-balance neutral point.  The camera only accepts a file made from one of
its own images (serial number check), so everything except the neutral point is copied from the
source image.  Kodak's quirks are kept: the entry counts are fixed (tags missing from the source
are skipped without lowering the count), values of 4 bytes or more are stored out of line, data
starts at 0x230, the KodakIFD pointer advances the data cursor by its count, and the Kodak IFD's
next-IFD offset points just past itself.
"""
import struct
from pathlib import Path

IFD0_TAGS = (0x10E, 0x10F, 0x110, 0x131, 0x132, 0x13B, 0x827D, 0x8290)
KODAK_IFD = 0x8290
KODAK_TAGS = (0x3E9, 0x40C, 0x7D0, 0x7D1, 0x7D2, 0x7D3, 0x7D4, 0x7DA, 0x7DB, 0x7DC, 0x7DD, 0x7DE,
              0x7E4, 0x7E5, 0x7E6, 0x7E7, 0x7E8, 0x834, 0x835, 0x836, 0x837, 0x838, 0x83E, 0x83F,
              0x840, 0x841, 0x842, 0x848, 0x849, 0x84A, 0x84B, 0x84C, 0x898, 0x899, 0x89A, 0x89B, 0x89C)
BALANCE_POINT = 0x40C
DATA_START = 0x230
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_ELEM = {3: "H", 8: "h", 4: "I", 9: "i", 11: "f", 5: "I", 10: "i", 12: "d"}   # rationals: 2 ints


class WBError(ValueError):
    pass


def _read_ifd(data, bo, offset):
    """{tag: (type, count, value bytes in file byte order)}"""
    (n,) = struct.unpack_from(bo + "H", data, offset)
    out = {}
    for i in range(n):
        pos = offset + 2 + 12 * i
        tag, typ, count = struct.unpack_from(bo + "HHI", data, pos)
        size = TYPE_SIZE.get(typ, 0) * count
        if size <= 4:
            raw = data[pos + 8:pos + 8 + size]
        else:
            (ptr,) = struct.unpack_from(bo + "I", data, pos + 8)
            raw = data[ptr:ptr + size]
        out[tag] = (typ, count, raw)
    return out


def _to_big_endian(typ, raw, bo):
    if bo == ">" or typ not in _ELEM:
        return raw
    code = _ELEM[typ]
    n = len(raw) // struct.calcsize(code)
    return struct.pack(">%d%s" % (n, code), *struct.unpack("<%d%s" % (n, code), raw))


def read_source(raw_path):
    """(IFD0 entries, Kodak IFD entries) of a DCS raw file."""
    data = Path(raw_path).read_bytes()
    bo = {b"MM": ">", b"II": "<"}.get(data[:2])
    if bo is None:
        raise WBError("not a TIFF-based DCS raw file")
    (first,) = struct.unpack_from(bo + "I", data, 4)
    ifd0 = _read_ifd(data, bo, first)
    if KODAK_IFD not in ifd0:
        raise WBError("no Kodak IFD")
    (kofs,) = struct.unpack(bo + "I", ifd0[KODAK_IFD][2][:4])
    kodak = _read_ifd(data, bo, kofs)
    conv = lambda d: {t: (ty, c, _to_big_endian(ty, raw, bo)) for t, (ty, c, raw) in d.items()}
    return conv(ifd0), conv(kodak)


def neutral_point(wb, reference=None):
    """Neutral (R, G, B) camera values that the multipliers ``wb`` balance to grey, on the scale of
    the source file's own balance point when it has one."""
    r, g, b = (float(v) for v in wb)
    green = reference[1] if reference and reference[1] else 128
    return tuple(max(1, min(65535, round(green * g / m))) for m in (r, g, b))


def build(raw_path, neutral):
    ifd0, kodak = read_source(raw_path)
    buf = bytearray(0x10000)
    buf[0:4] = b"MM\x00\x2a"
    struct.pack_into(">I", buf, 4, 8)
    pos, data = 8, DATA_START

    def entry(tag, typ, count, raw):
        nonlocal pos, data
        struct.pack_into(">HHI", buf, pos, tag, typ, count)
        if len(raw) < 4:
            buf[pos + 8:pos + 8 + len(raw)] = raw
        else:
            struct.pack_into(">I", buf, pos + 8, data)
            buf[data:data + len(raw)] = raw
            data += len(raw)
        pos += 12

    struct.pack_into(">H", buf, pos, len(IFD0_TAGS))
    pos += 2
    for tag in IFD0_TAGS:
        if tag == KODAK_IFD:
            typ, count, _ = ifd0[tag]
            struct.pack_into(">HHII", buf, pos, tag, typ, count, pos + 16)
            data += count
            pos += 12
        elif tag in ifd0 and ifd0[tag][0] in TYPE_SIZE:
            entry(tag, *ifd0[tag])
    struct.pack_into(">I", buf, pos, 0)        # IFD0 has no next IFD
    pos += 4
    struct.pack_into(">H", buf, pos, len(KODAK_TAGS))
    pos += 2
    for tag in KODAK_TAGS:
        if tag == BALANCE_POINT:
            entry(tag, 3, 3, struct.pack(">3H", *neutral))
        elif tag in kodak and kodak[tag][0] in TYPE_SIZE:
            entry(tag, *kodak[tag])
    struct.pack_into(">I", buf, pos, pos + 4)  # as written by Photo Desk: an empty IFD follows
    return bytes(buf[:data])


def write(path, raw_path, wb):
    """Save the click balance ``wb`` (camera RGB multipliers) of ``raw_path`` as a .wb file."""
    ifd0, kodak = read_source(raw_path)
    ref = None
    if BALANCE_POINT in kodak and kodak[BALANCE_POINT][0] == 3 and kodak[BALANCE_POINT][1] == 3:
        ref = struct.unpack(">3H", kodak[BALANCE_POINT][2])
    Path(path).write_bytes(build(raw_path, neutral_point(wb, ref)))


def read_neutral(path):
    """Neutral point stored in a .wb file."""
    _, kodak = read_source(path)
    return struct.unpack(">3H", kodak[BALANCE_POINT][2])
