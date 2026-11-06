import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from photodesk import camera, pipeline


def _tiff_entry(byteorder, tag, typ, count, value):
    if typ == 2:
        raw = value
        field = raw.ljust(4, b"\0") if count <= 4 else struct.pack(byteorder + "I", value)
    elif typ == 3:
        raw = struct.pack(byteorder + "H", value)
        field = raw.ljust(4, b"\0")
    else:
        field = struct.pack(byteorder + "I", value)
    return struct.pack(byteorder + "HHI", tag, typ, count) + field


def _capture_tiff(path, byteorder="<", *, root_fields=None, settings=None,
                  settings_count=None, settings_payload=None, include_kodak=True):
    """Write a minimal TIFF with standard IFD0 fields and an optional Kodak IFD."""
    root_fields = list(root_fields or [])
    root_fields.extend([
        (256, 4, 1, 1), (257, 4, 1, 1), (258, 3, 1, 8), (259, 3, 1, 1),
        (262, 3, 1, 1), (273, 4, 1, 0), (277, 3, 1, 1), (278, 4, 1, 1), (279, 4, 1, 1),
    ])
    marker = b"II*\0" if byteorder == "<" else b"MM\0*"
    root_entries = list(root_fields)
    if include_kodak:
        root_entries.append((camera.KODAK_IFD, 4, 1, None))
    root_entries.sort(key=lambda entry: entry[0])

    root_offset = 8
    root_end = root_offset + 2 + 12 * len(root_entries) + 4
    root_data = bytearray()
    packed_root = []
    for tag, typ, count, value in root_entries:
        if tag == camera.KODAK_IFD:
            packed_root.append((tag, typ, count, 0))
        elif tag == 273:
            packed_root.append((tag, typ, count, 0))
        elif typ == 2 and count > 4:
            offset = root_end + len(root_data)
            packed_root.append((tag, typ, count, offset))
            root_data.extend(value)
        else:
            packed_root.append((tag, typ, count, value))

    kodak_offset = root_end + len(root_data)
    kodak = bytearray()
    kodak_data = b""
    if include_kodak:
        has_settings = settings is not None or settings_payload is not None or settings_count is not None
        payload = settings if settings_payload is None else settings_payload
        payload = b"" if payload is None else payload
        count = len(payload) if settings_count is None else settings_count
        entry_count = int(has_settings)
        kodak.extend(struct.pack(byteorder + "H", entry_count))
        if has_settings:
            if count <= 4:
                field_value = payload
            else:
                field_value = kodak_offset + 2 + 12 * entry_count + 4
                kodak_data = payload
            # Kodak tag 1009 is the only optional entry in this fixture.
            field = payload.ljust(4, b"\0") if count <= 4 else struct.pack(byteorder + "I", field_value)
            kodak.extend(struct.pack(byteorder + "HHI", 1009, 2, count) + field)
        kodak.extend(b"\0" * 4)

    root = bytearray(struct.pack(byteorder + "H", len(packed_root)))
    for tag, typ, count, value in packed_root:
        if tag == camera.KODAK_IFD:
            value = kodak_offset
        elif tag == 273:
            value = kodak_offset + len(kodak) + len(kodak_data)
        root.extend(_tiff_entry(byteorder, tag, typ, count, value))
    root.extend(b"\0" * 4)
    path.write_bytes(marker + struct.pack(byteorder + "I", root_offset) + root + root_data + kodak
                     + kodak_data + b"\0")


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_read_capture_metadata_ifd0_and_kodak_ascii_in_both_endian_files(tmp_path, byteorder):
    settings = (b"Exposure Bias: +0.0 EV\nISO Speed: 200\n"
                b" (Auto strength =10)\nAF: On\n\0")
    fields = [
        (0x131, 2, 6, b"5.4.1\0"),
        (0x132, 2, 20, b"2026:10:05 10:00:00\0"),
        (0x827D, 2, 12, b"PSLRN-24584\0"),
        (0x9003, 2, 20, b"2026:10:05 11:13:25\0"),
        (0x9211, 4, 1, 2034),
    ]
    path = tmp_path / "capture.dcr"
    _capture_tiff(path, byteorder, root_fields=fields, settings=settings)

    metadata = camera.read_capture_metadata(path)
    assert metadata == {
        "serial": "PSLRN-24584",
        "firmware": "5.4.1",
        "image_number": 2034,
        "capture_timestamp": "2026:10:05 11:13:25",
        "camera_settings": ("Exposure Bias: +0.0 EV", "ISO Speed: 200", " (Auto strength =10)", "AF: On"),
    }
    assert camera.read_kodak_tags(path, (1009,)) == {}
    # Camera screenshots verify Kodak's native CameraSettingString rows exactly as recorded.
    assert camera.read_kodak_tags(path, (1009,), expected_types={1009: 2},
                                  max_count=len(settings))[1009] == tuple(settings)


def test_read_capture_metadata_uses_datetime_when_original_is_missing(tmp_path):
    path = tmp_path / "datetime.dcr"
    _capture_tiff(path, root_fields=[(0x132, 2, 20, b"2026:10:05 11:13:25\0")], include_kodak=False)
    assert camera.read_capture_metadata(path) == {"capture_timestamp": "2026:10:05 11:13:25"}


@pytest.mark.parametrize("count,payload", [(8, b"short"), (65537, b"")])
def test_read_capture_metadata_skips_truncated_or_oversized_kodak_text(tmp_path, count, payload):
    path = tmp_path / "malformed.dcr"
    _capture_tiff(path, settings_count=count, settings_payload=payload)
    assert 1009 not in camera.read_kodak_tags(path, (1009,), expected_types={1009: 2}, max_count=65536)
    assert "camera_settings" not in camera.read_capture_metadata(path)


def test_read_capture_metadata_omits_missing_fields_and_non_tiff_files(tmp_path):
    empty_tiff = tmp_path / "empty.dcr"
    _capture_tiff(empty_tiff, include_kodak=False)
    assert camera.read_capture_metadata(empty_tiff) == {}

    not_tiff = tmp_path / "not-a-tiff.dcr"
    not_tiff.write_bytes(b"not a TIFF")
    assert camera.read_capture_metadata(not_tiff) == {}
    assert camera.read_capture_metadata(tmp_path / "missing.dcr") == {}


@pytest.mark.parametrize("header", [b"II*\x00", b"MM\x00*", b"II*\x00\x08\x00"])
def test_capture_metadata_skips_truncated_tiff_header(tmp_path, header):
    path = tmp_path / "truncated.dcr"
    path.write_bytes(header)
    assert camera.read_capture_metadata(path) == {}


def test_decode_merges_capture_metadata_without_changing_raw_processing(tmp_path, monkeypatch):
    path = tmp_path / "decode.dcr"
    _capture_tiff(path, root_fields=[(0x827D, 2, 12, b"PSLRN-24584\0")], include_kodak=False)

    class FakeRaw:
        sizes = SimpleNamespace(width=4, height=3, flip=0)
        daylight_whitebalance = [1.0, 1.0, 1.0, 1.0]
        camera_whitebalance = [1.2, 1.0, 1.3, 1.0]
        rgb_xyz_matrix = np.eye(3)
        white_level = 65535
        other = SimpleNamespace(iso_speed=100, shutter_speed=0.01, aperture=4.0, focal_length=50.0,
                                timestamp=None, artist="")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def postprocess(self, **_):
            return np.full((3, 4, 3), 32768, dtype=np.uint16)

    monkeypatch.setattr(pipeline.camera, "read_make_model", lambda _: ("Kodak", "DCS 999"))
    monkeypatch.setattr(pipeline.rawpy, "imread", lambda _: FakeRaw())
    raw = pipeline.decode(path, half_size=True)

    assert raw.metadata["serial"] == "PSLRN-24584"
    assert raw.metadata["width"] == 4 and raw.metadata["height"] == 3
    np.testing.assert_allclose(raw.camera_rgb, np.full((3, 4, 3), 32768 / 65535, dtype=np.float32))


def test_capture_metadata_from_optional_slrn_sample():
    sample = os.environ.get("PHOTODESK_SLRN_SAMPLE")
    if not sample or not Path(sample).is_file():
        pytest.skip("PHOTODESK_SLRN_SAMPLE not available")
    metadata = camera.read_capture_metadata(sample)
    assert metadata["serial"] == "PSLRN-24584"
    assert metadata["firmware"] == "5.4.1"
    assert metadata["image_number"] == 2034
    assert metadata["capture_timestamp"] == "2026:10:05 11:13:25"
    assert "White bal: Standard Daylight 5500" in metadata["camera_settings"]
