"""IPTC/Job Tracker data (IPTC-IIM record 2), as edited in Photo Desk's IPTC dialog.

Data are kept as {field key: str}; fields marked repeatable in the dialog
("Supplemental Category*", "Keywords*") hold semicolon separated items.
.ipt files contain the raw record 2 datasets, as written by Photo Desk's
"Save As..." and loaded by "Load...".
"""
import struct
from pathlib import Path

# (key, dataset number, max length, repeatable) in the order of the IPTC dialog.
FIELDS = (
    ("object_name", 5, 64, False),
    ("city", 90, 32, False),
    ("province_state", 95, 32, False),
    ("country", 101, 64, False),
    ("date_created", 55, 8, False),           # CCYYMMDD
    ("category", 15, 3, False),
    ("urgency", 10, 1, False),
    ("supplemental_category", 20, 32, True),
    ("byline", 80, 32, False),
    ("byline_title", 85, 32, False),
    ("credit", 110, 32, False),
    ("original_transmission_reference", 103, 32, False),
    ("source", 115, 32, False),
    ("caption_writer", 122, 32, False),
    ("keywords", 25, 64, True),
    ("caption", 120, 2000, False),
    ("headline", 105, 256, False),
    ("copyright", 116, 128, False),
    ("special_instructions", 40, 256, False),
)
_BY_NUMBER = {n: (k, rep) for k, n, _, rep in FIELDS}
URGENCY = ("", "1 (Most)", "2", "3", "4", "5 (Normal)", "6", "7", "8 (Least)")


def encode(data):
    out = bytearray(b"\x1c\x02\x00\x00\x02\x00\x04")  # record version 4
    for key, number, limit, repeat in FIELDS:
        value = (data.get(key) or "").strip()
        if not value:
            continue
        items = [v.strip() for v in value.split(";") if v.strip()] if repeat else [value]
        for item in items:
            b = item.encode("utf-8")[:limit]
            out += struct.pack(">BBBH", 0x1C, 2, number, len(b)) + b
    return bytes(out)


def decode(blob):
    data = {}
    pos = 0
    while pos + 5 <= len(blob) and blob[pos] == 0x1C:
        rec, number, size = blob[pos + 1], blob[pos + 2], struct.unpack_from(">H", blob, pos + 3)[0]
        pos += 5
        if size & 0x8000:  # extended length
            n = size & 0x7FFF
            size = int.from_bytes(blob[pos:pos + n], "big")
            pos += n
        value = blob[pos:pos + size]
        pos += size
        if rec != 2 or number not in _BY_NUMBER:
            continue
        key, repeat = _BY_NUMBER[number]
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            text = value.decode("latin1")
        data[key] = f"{data[key]}; {text}" if repeat and data.get(key) else text
    return data


def load_ipt(path):
    return decode(Path(path).read_bytes())


def save_ipt(path, data):
    Path(path).write_bytes(encode(data))


def insert_into_jpeg(path, blob):
    """Add an APP13 Photoshop 3.0 IRB carrying the IPTC record after the APP0/APP1 segments."""
    path = Path(path)
    jpg = path.read_bytes()
    irb = b"8BIM" + struct.pack(">H", 0x0404) + b"\0\0" + struct.pack(">I", len(blob)) + blob
    if len(blob) % 2:
        irb += b"\0"
    payload = b"Photoshop 3.0\0" + irb
    seg = b"\xff\xed" + struct.pack(">H", len(payload) + 2) + payload
    pos = 2
    while pos + 4 <= len(jpg) and jpg[pos] == 0xFF and jpg[pos + 1] in (0xE0, 0xE1, 0xE2):
        pos += 2 + struct.unpack_from(">H", jpg, pos + 2)[0]
    path.write_bytes(jpg[:pos] + seg + jpg[pos:])
