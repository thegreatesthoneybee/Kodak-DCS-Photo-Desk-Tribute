"""Tests run against the original installer and sample raw when available.

Set PHOTODESK_INSTALLER / PHOTODESK_SAMPLE to override the default locations.
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest

from photodesk import camera, export, icc, mscab, pipeline, profiles

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("PHOTODESK_INSTALLER", ROOT / "vendor/original/Photo_Desk_V_4_3_0.exe"))
SAMPLE = Path(os.environ.get("PHOTODESK_SAMPLE", ROOT / "raw/4H6B9130.DCR"))

need_installer = pytest.mark.skipif(not INSTALLER.is_file(), reason="Photo Desk installer not available")
need_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="sample DCR not available")


@pytest.fixture(scope="session")
def profile_dir(tmp_path_factory):
    if not INSTALLER.is_file():
        pytest.skip("Photo Desk installer not available")
    d = tmp_path_factory.mktemp("profiles")
    profiles.install_from_installer(INSTALLER, d)
    return d


@pytest.fixture(scope="session")
def raw():
    if not SAMPLE.is_file():
        pytest.skip("sample DCR not available")
    return pipeline.decode(SAMPLE, half_size=True)


def test_mscab_stored_roundtrip():
    payload = b"hello cabinet" * 10
    name = b"a.txt\x00"
    # Minimal single-folder, single-file, uncompressed cabinet.
    hdr_len, folder_len, file_len = 36, 8, 16 + len(name)
    data_off = hdr_len + folder_len + file_len
    size = data_off + 8 + len(payload)
    import struct
    cab = (b"MSCF" + b"\0" * 4 + struct.pack("<I", size) + b"\0" * 4 + struct.pack("<I", hdr_len + folder_len)
           + b"\0" * 4 + bytes([3, 1]) + struct.pack("<HHHHH", 1, 1, 0, 0, 0)
           + struct.pack("<IHH", data_off, 1, 0)
           + struct.pack("<IIHHHH", len(payload), 0, 0, 0, 0, 0x20) + name
           + struct.pack("<IHH", 0, len(payload), len(payload)) + payload)
    blob = b"junk" + cab + b"tail"
    found = list(mscab.find_cabinets(blob))
    assert [off for off, _ in found] == [4]
    assert mscab.extract(found[0][1]) == {"a.txt": payload}


def test_camera_mapping():
    assert camera.source_profile("ProBack645", "Tungsten") == "DCSPB645TungstenSource.icm"
    assert camera.source_profile("DCS Pro SLR/n", "Daylight") == "DCSProSLRnDaylight.icm"
    assert camera.source_profile("DCS Pro 14N", "Flash") == "DCSPro14Flash.icm"
    assert camera.source_profile("DCS760C", "Fluorescent") == "DCS760CFluorescentSource.icm"
    assert camera.source_profile("Canon EOS 5D", "Daylight") is None
    with pytest.raises(ValueError):
        camera.source_profile("ProBack645", "Moonlight")


def test_lab_xyz_roundtrip():
    lab = np.array([[50.0, 20.0, -30.0], [90.0, -5.0, 5.0], [10.0, 0.0, 0.0]], np.float32)
    np.testing.assert_allclose(icc.xyz_to_lab(icc.lab_to_xyz(lab)), lab, atol=1e-3)
    np.testing.assert_allclose(icc.lab_decode(icc.lab_encode(lab)), lab, atol=1e-3)


def test_erimm_encoding_reference_points():
    # ISO 22028-3: E_clip encodes to 1.0, E_min*e is the toe/log break point.
    assert pipeline.erimm_encode(np.float32(10 ** 2.5)) == pytest.approx(1.0, abs=1e-6)
    assert pipeline.erimm_encode(np.float32(0)) == 0
    assert pipeline.erimm_encode(np.float32(0.18)) == pytest.approx((np.log10(0.18) + 3) / 5.5, abs=1e-5)


def test_settings_json_roundtrip(tmp_path):
    p = tmp_path / "x.DCR"
    p.write_bytes(b"")
    s = pipeline.Settings(look="DCSPortraitLook.icm", wb_mode="custom", wb_custom=(1.2, 1.0, 1.5),
                          exposure=0.7, rotation=5)
    pipeline.save_settings(p, s)
    loaded = pipeline.load_settings(p)
    assert loaded.look == "DCSPortraitLook.icm"
    assert loaded.wb_custom == (1.2, 1.0, 1.5)
    assert loaded.rotation == 1
    assert pipeline.load_settings(tmp_path / "other.DCR", "Kodak").wb_mode == "daylight"
    assert pipeline.load_settings(tmp_path / "other.NEF", "Nikon").wb_mode == "as_shot"


@need_installer
def test_installer_profiles(profile_dir):
    names = {p.name for p in profile_dir.glob("*.icm")}
    assert len(names) == 57
    assert {"ERIMM_PCS3.icm", "DCSPB645DaylightSource.icm", "DCSProductLook.icm"} <= names
    prof = icc.Profile(profile_dir / "DCSPB645DaylightSource.icm")
    assert prof.description == "DCS PB645 Daylight Source Correction"


@need_installer
def test_source_profile_neutral_axis(profile_dir):
    lut = icc.Profile(profile_dir / "DCSPB645DaylightSource.icm").lut()
    lab = icc.lab_decode(lut(np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]], np.float32)))
    assert np.all(np.abs(lab[:, 1:]) < 1.0)          # equal camera RGB stays neutral
    assert lab[0, 0] > lab[1, 0]                     # monotonic lightness


@need_installer
def test_look_mid_grey(profile_dir):
    look = icc.Profile(profile_dir / "DCSProductLook.icm").lut()
    grey = pipeline.erimm_encode(np.full((1, 3), 0.18, np.float32))
    L = icc.lab_decode(look(grey))[0, 0]
    assert 45 < L < 65


@need_installer
def test_icc_output_profiles_parse():
    from PIL import ImageCms
    for space in pipeline.OUTPUT_SPACES:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(export.icc_profile(space)))
        assert ImageCms.getProfileDescription(prof).strip() == space
        ImageCms.buildTransform(prof, ImageCms.createProfile("sRGB"), "RGB", "RGB")


@need_sample
def test_decode_sample(raw):
    assert raw.make == "Kodak" and raw.model == "ProBack645"
    assert raw.camera_rgb.dtype == np.float32
    assert raw.camera_rgb.shape == (2040, 2040, 3)
    assert 0 < raw.camera_rgb.max() <= 1


@need_sample
@need_installer
def test_render_sample(raw, profile_dir):
    s = pipeline.default_settings(raw.make)
    img = pipeline.render(raw, s, "sRGB", profile_dir)
    assert img.dtype == np.float32 and img.shape == (2040, 2040, 3)
    assert 0.15 < float(np.median(img)) < 0.5
    brighter = pipeline.render(raw, pipeline.Settings(**{**s.to_json(), "exposure": 1.0}), "sRGB", profile_dir)
    assert np.median(brighter) > np.median(img)
    rotated = pipeline.render(raw, pipeline.Settings(**{**s.to_json(), "rotation": 1}), "sRGB", profile_dir)
    np.testing.assert_allclose(rotated, np.rot90(img, -1), atol=1e-6)


@need_sample
@need_installer
def test_daylight_wb_neutralises_grey_wall(raw, profile_dir):
    s = pipeline.Settings(wb_mode="daylight", sharpen=0, noise_chroma=0)
    img = pipeline.render(raw, s, "sRGB", profile_dir)
    h, w = img.shape[:2]
    patch = img[int(.37 * h) - 8:int(.37 * h) + 8, int(.18 * w) - 8:int(.18 * w) + 8].reshape(-1, 3).mean(0)
    lin = np.where(patch <= 0.04045, patch / 12.92, ((patch + 0.055) / 1.055) ** 2.4)
    lab = icc.xyz_to_lab(lin @ np.linalg.inv(pipeline.OUTPUT_SPACES["sRGB"][0]).T)
    assert np.hypot(lab[1], lab[2]) < 5


@need_sample
@need_installer
def test_export_formats(tmp_path, profile_dir):
    import tifffile
    from PIL import Image
    s = pipeline.Settings(wb_mode="daylight")
    jpg = export.process_file(SAMPLE, tmp_path, "JPEG", "sRGB", 90, 800, s, profile_dir)
    im = Image.open(jpg)
    assert im.size == (800, 800) and "icc_profile" in im.info
    assert im.getexif()[0x0110] == "ProBack645"
    tif = export.process_file(SAMPLE, tmp_path, "TIFF 16-bit", "ProPhoto RGB", 90, None, s, profile_dir)
    page = tifffile.TiffFile(tif).pages[0]
    assert page.dtype == np.uint16 and page.shape == (4080, 4080, 3)
    assert 34675 in page.tags
