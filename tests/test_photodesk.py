"""Tests run against the original installer and sample raw when available.

Set PHOTODESK_INSTALLER / PHOTODESK_SAMPLE to override the default locations.
"""
import io
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from photodesk import camera, export, icc, iptc, mscab, pipeline, profiles, ui_strings
from photodesk.peres import PEResources

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


def _srgb_lab(img, profile_dir, fx=.18, fy=.37):
    h, w = img.shape[:2]
    p = img[int(fy * h) - 8:int(fy * h) + 8, int(fx * w) - 8:int(fx * w) + 8].reshape(-1, 3).mean(0)
    lin = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
    m = icc.Profile(profile_dir / "sRGB Color Space Profile.icm").rgb_to_xyz_matrix()
    return icc.xyz_to_lab(lin @ m.T)


def test_mscab_stored_roundtrip():
    payload = b"hello cabinet" * 10
    name = b"a.txt\x00"
    hdr_len, folder_len, file_len = 36, 8, 16 + len(name)
    data_off = hdr_len + folder_len + file_len
    size = data_off + 8 + len(payload)
    cab = (b"MSCF" + b"\0" * 4 + struct.pack("<I", size) + b"\0" * 4 + struct.pack("<I", hdr_len + folder_len)
           + b"\0" * 4 + bytes([3, 1]) + struct.pack("<HHHHH", 1, 1, 0, 0, 0)
           + struct.pack("<IHH", data_off, 1, 0)
           + struct.pack("<IIHHHH", len(payload), 0, 0, 0, 0, 0x20) + name
           + struct.pack("<IHH", 0, len(payload), len(payload)) + payload)
    found = list(mscab.find_cabinets(b"junk" + cab + b"tail"))
    assert [off for off, _ in found] == [4]
    assert mscab.extract(found[0][1]) == {"a.txt": payload}


def test_camera_mapping_and_capabilities():
    assert camera.source_profile("ProBack645", "Tungsten") == "DCSPB645TungstenSource.icm"
    assert camera.source_profile("DCS Pro SLR/n", "Daylight") == "DCSProSLRnDaylight.icm"
    assert camera.source_profile("DCS Pro 14N", "Flash") == "DCSPro14Flash.icm"
    assert camera.source_profile("Canon EOS 5D", "Daylight") is None
    with pytest.raises(ValueError):
        camera.source_profile("ProBack645", "Moonlight")
    pb = camera.capabilities("ProBack645")
    assert pb.exposure_range == 1.0 and "custom" not in pb.balances and "basic" in pb.noise_types
    slr = camera.capabilities("DCS Pro SLR/n")
    assert slr.exposure_range == 2.0 and "expert" in slr.noise_types and slr.lens_optimization
    assert not camera.capabilities("DCS Pro 14N").lens_optimization


def test_lab_xyz_roundtrip():
    lab = np.array([[50.0, 20.0, -30.0], [90.0, -5.0, 5.0], [10.0, 0.0, 0.0]], np.float32)
    np.testing.assert_allclose(icc.xyz_to_lab(icc.lab_to_xyz(lab)), lab, atol=1e-3)
    np.testing.assert_allclose(icc.lab_decode(icc.lab_encode(lab)), lab, atol=1e-3)


def test_erimm_encoding_reference_points():
    assert pipeline.erimm_encode(np.float32(10 ** 2.5)) == pytest.approx(1.0, abs=1e-6)
    assert pipeline.erimm_encode(np.float32(0)) == 0
    assert pipeline.erimm_encode(np.float32(0.18)) == pytest.approx((np.log10(0.18) + 3) / 5.5, abs=1e-5)


def test_settings_json_roundtrip(tmp_path):
    p = tmp_path / "x.DCR"
    p.write_bytes(b"")
    s = pipeline.Settings(look="DCSPortraitLook.icm", balance="click", click_wb=(1.2, 1.0, 1.5),
                          exposure=0.7, rotation=5, crop=(0.1, 0.2, 0.5, 0.9), tagged=True,
                          annotation="note", iptc={"city": "Seoul"}, tone=(-8.0, 0.5, 3.0))
    pipeline.save_settings(p, s)
    loaded = pipeline.load_settings(p)
    assert loaded.click_wb == (1.2, 1.0, 1.5) and loaded.rotation == 1
    assert loaded.crop == (0.1, 0.2, 0.5, 0.9) and loaded.tagged and loaded.iptc == {"city": "Seoul"}
    assert loaded.tone == (-8.0, 0.5, 3.0)
    assert pipeline.load_settings(tmp_path / "other.DCR").balance == "preset"


def test_tone_validation_and_mapping():
    assert pipeline.validate_tone(pipeline.TONE_DEFAULT) is None
    assert pipeline.validate_tone((-1.0, 0.0, 3.4)) == "Shadow value is too high."
    assert pipeline.validate_tone((-9.5, 2.0, 3.4)) == "Midtone value is too high."
    ev = np.array([-9.5, -1.0, 3.4])
    np.testing.assert_allclose(pipeline.tone_map_ev(ev, (-9.5, -1.0, 3.4)), [-9.5, 0.0, 3.4])


def test_crop_rotation_and_sequence_names():
    assert pipeline.rotate_crop((0.0, 0.0, 0.5, 0.25), 1) == (0.75, 0.0, 1.0, 0.5)
    assert pipeline.rotate_crop((0.1, 0.2, 0.3, 0.4), 4) == (0.1, 0.2, 0.3, 0.4)
    assert export.sequence_names("shoes", 4, 5, 3) == ["shoes0005", "shoes0006", "shoes0007"]
    assert export.sequence_names("a*b", 2, 9, 2) == ["a09b", "a10b"]


def test_export_size_styles():
    o = export.OutputOptions(resize=True, scale_style="percent", percent=50)
    assert export.export_size(4000, 3000, o) == (2000, 1500, 2000, 1500)
    o = export.OutputOptions(resize=True, scale_style="fill", width=1000, height=1000)
    assert export.export_size(4000, 3000, o)[2:] == (1000, 1000)
    o.scale_style = "fit_trim"
    assert export.export_size(4000, 3000, o) == (1000, 750, 1000, 750)
    o.scale_style = "fit_fill"
    img = export.apply_size(np.zeros((300, 400, 3), np.float32), o)
    assert img.shape == (1000, 1000, 3) and img[0, 0, 0] == 1.0 and img[500, 500, 0] == 0.0


def test_iptc_roundtrip(tmp_path):
    data = {"object_name": "Shoot", "keywords": "a; b", "city": "서울", "caption": "x" * 100}
    assert iptc.decode(iptc.encode(data)) == data
    iptc.save_ipt(tmp_path / "x.ipt", data)
    assert iptc.load_ipt(tmp_path / "x.ipt") == data


def test_translator():
    table = ("LOCATION\tKEY\tDESCRIPTION\t" + "\t".join(ui_strings.LANGUAGES) + "\r\n"
             + "File Menu\tk\td\t&Open...\\tCtrl+O" + "\t" * 6 + "열기(&O)...\\tCtrl+O\t\t\t\r\n"
             + "X\tk\td\tCrop Aspect Ratio:" + "\t" * 6 + "트리밍 비율:\t\t\t\r\n").encode("utf-16")
    tr = ui_strings.Translator(table, "Korean")
    assert tr("&Open...\tCtrl+O") == "열기(&O)...\tCtrl+O"
    assert tr("Crop Aspect Ratio") == "트리밍 비율"
    assert tr("Untranslated") == "Untranslated"
    assert ui_strings.Translator(table, "English")("&Open...\tCtrl+O") == "&Open...\tCtrl+O"


@need_installer
def test_installer_profiles_and_ui(profile_dir):
    names = {p.name for p in profile_dir.glob("*.icm")}
    assert len(names) == 57
    assert {"ERIMM_PCS3.icm", "DCSPB645DaylightSource.icm", "DCSProductLook.icm"} <= names
    assert icc.Profile(profile_dir / "DCSPB645DaylightSource.icm").description == "DCS PB645 Daylight Source Correction"
    res = PEResources((profiles.ui_dir(profile_dir) / "PhotoDsk.exe").read_bytes())
    assert res.strings()[59148] == "%d images,  %d selected"
    assert res.toolbar(128)[0] == (16, 15) and res.toolbar(128)[1][:2] == [57601, 32868]
    panel = res.dialog(103)
    assert panel.size == (156, 407) and panel.control(1172).cls == "ComboBox"
    assert res.menu(128)[0][1] == "&File"
    tr = ui_strings.Translator((profiles.ui_dir(profile_dir) / "PhotoDeskUIStrings.u16").read_bytes(), "Korean")
    assert tr("&File") == "파일(&F)"


@need_installer
def test_source_profile_neutral_axis(profile_dir):
    lut = icc.Profile(profile_dir / "DCSPB645DaylightSource.icm").lut()
    lab = icc.lab_decode(lut(np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]], np.float32)))
    assert np.all(np.abs(lab[:, 1:]) < 1.0)
    assert lab[0, 0] > lab[1, 0]


@need_installer
def test_working_spaces(profile_dir):
    spaces = dict((desc, path) for path, desc in pipeline.working_spaces(profile_dir))
    assert {"sRGB IEC61966-2.1", "Kodak ProPhoto RGB", "Kodak DCS RGB"} <= set(spaces)
    enc = icc.Profile(spaces["sRGB IEC61966-2.1"]).encoder()
    white = enc(icc.D50[None].astype(np.float32))[0]
    np.testing.assert_allclose(white, 1.0, atol=2e-3)


@need_sample
def test_decode_sample(raw):
    assert raw.make == "Kodak" and raw.model == "ProBack645"
    assert raw.camera_rgb.shape == (2040, 2040, 3)
    assert 0 < raw.camera_rgb.max() <= 1


@need_sample
@need_installer
def test_preset_balance_neutralises_grey_wall(raw, profile_dir):
    s = pipeline.Settings(sharpen_amount=0)
    img = pipeline.render(raw, s, profile_directory=profile_dir)
    lab = _srgb_lab(img, profile_dir)
    assert np.hypot(lab[1], lab[2]) < 5
    warm = _srgb_lab(pipeline.render(raw, s.copy(white_point=7000), profile_directory=profile_dir), profile_dir)
    assert warm[2] > lab[2] + 5
    red = _srgb_lab(pipeline.render(raw, s.copy(color_balance=(60, 0, 0)), profile_directory=profile_dir), profile_dir)
    assert red[1] > lab[1] + 5 and abs(red[0] - lab[0]) < 3


@need_sample
@need_installer
def test_render_options(raw, profile_dir):
    s = pipeline.Settings(sharpen_amount=0)
    img = pipeline.render(raw, s, profile_directory=profile_dir)
    assert img.dtype == np.float32 and img.shape == (2040, 2040, 3)
    brighter = pipeline.render(raw, s.copy(exposure=1.0), profile_directory=profile_dir)
    assert np.median(brighter) > np.median(img)
    mid = pipeline.render(raw, s.copy(tone=(-9.5, -1.0, 3.4)), profile_directory=profile_dir)
    assert np.median(mid) > np.median(img)
    rotated = pipeline.render(raw, s.copy(rotation=1), profile_directory=profile_dir)
    np.testing.assert_allclose(rotated, np.rot90(img, -1), atol=1e-6)
    cropped = pipeline.render(raw, s.copy(crop=(0.25, 0.25, 0.75, 0.5)), profile_directory=profile_dir)
    assert cropped.shape == (510, 1020, 3)
    fast = pipeline.render(raw, pipeline.Settings(noise_type="advanced"), profile_directory=profile_dir,
                           quality="faster")
    assert fast.shape == img.shape
    assert -1.0 <= pipeline.auto_exposure(raw, s) <= 1.0


@need_sample
@need_installer
def test_export_formats(tmp_path, profile_dir):
    import tifffile
    from PIL import Image
    s = pipeline.Settings(iptc={"object_name": "Test"}, annotation="hello")
    opts = export.OutputOptions(resize=True, scale_style="fit_trim", width=800, height=800, jpeg_quality="better")
    jpg = export.process_file(SAMPLE, tmp_path / "a.jpg", "jpeg", opts, s, profile_dir)
    im = Image.open(jpg)
    assert im.size == (800, 800) and "icc_profile" in im.info and im.getexif()[0x0110] == "ProBack645"
    assert b"Photoshop 3.0" in jpg.read_bytes()
    opts = export.OutputOptions(tiff_16bit=True, working_space="ProPhotoRGB.icm")
    tif = export.process_file(SAMPLE, tmp_path / "b.tif", "tiff", opts, s, profile_dir)
    page = tifffile.TiffFile(tif).pages[0]
    assert page.dtype == np.uint16 and page.shape == (4072, 4072, 3) or page.shape == (4080, 4080, 3)
    assert 34675 in page.tags and 33723 in page.tags
    lin = export.process_file(SAMPLE, tmp_path / "c.tif", "tiff", export.OutputOptions(tiff_linear=True), s, profile_dir)
    assert 34675 not in tifffile.TiffFile(lin).pages[0].tags
    asis = export.process_file(SAMPLE, tmp_path / "d.DCR", "asis", None, s, profile_dir)
    assert asis.read_bytes() == SAMPLE.read_bytes() and pipeline.load_settings(asis).annotation == "hello"
    # Finished files: only rotation and crop apply.
    fin = export.process_file(jpg, tmp_path / "e.jpg", "jpeg", export.OutputOptions(),
                              pipeline.Settings(rotation=1, crop=(0, 0, 0.5, 1)), profile_dir)
    assert Image.open(fin).size == (400, 800)


def test_sheet_undo_redo_rename_paste(tmp_path):
    from PIL import Image
    from photodesk.gui.model import Sheet
    for n in ("b.jpg", "a.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / n)
    sheet = Sheet(tmp_path)
    a, b = sheet.entries
    assert a.path.name == "a.jpg" and not a.is_raw
    sheet.set_fields([a, b], "tag", tagged=True)
    sheet.set_fields([a], "rotate", rotation=1)
    assert sheet.undo_text_id() == 62103
    sheet.undo()
    assert a.settings.rotation == 0 and a.settings.tagged
    assert sheet.redo_text_id() == 62104
    sheet.redo()
    assert a.settings.rotation == 1
    sheet.paste([b], a.settings, ("rotation",))
    assert b.settings.rotation == 1
    assert sheet.save([a]) == [] and not pipeline.sidecar_path(a.path).is_file()   # finished file: not saved
    pipeline.save_settings(a.path, a.settings)
    sheet.rename([a, b], ["x01", "x02"])
    assert {e.path.name for e in sheet.entries} == {"x01.jpg", "x02.jpg"}
    assert pipeline.sidecar_path(tmp_path / "x01.jpg").is_file()
    sheet.undo()
    assert {e.path.name for e in sheet.entries} == {"a.jpg", "b.jpg"}
    gone = sheet.delete([b], lambda p: p.unlink() or True)
    assert gone == [b] and not (tmp_path / "b.jpg").exists()


def test_sheet_drop_copy_move(tmp_path):
    from PIL import Image
    from photodesk.gui.model import Sheet
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(), dst.mkdir()
    for n in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(src / n)
    (src / "a01.wav").write_bytes(b"RIFF")
    Image.new("RGB", (8, 8), "red").save(dst / "b.jpg")
    s = Sheet(src)
    s.set_fields(s.entries[:1], "tag", tagged=True)
    pipeline.save_settings(s.entries[0].path, s.entries[0].settings)
    sheet = Sheet(dst)
    asked = []
    copied = sheet.transfer([src / "a.jpg", src / "b.jpg"], False, lambda d: asked.append(d.name) or False)
    assert [p.name for p in copied] == ["a.jpg"] and asked == ["b.jpg"]
    assert (src / "a.jpg").exists() and (dst / "a01.wav").exists()
    assert sheet.find(dst / "a.jpg").settings.tagged
    moved = sheet.transfer([src / "b.jpg"], True, lambda d: True)
    assert [p.name for p in moved] == ["b.jpg"] and not (src / "b.jpg").exists()
    assert [e.path.name for e in sheet.entries] == ["a.jpg", "b.jpg"]
