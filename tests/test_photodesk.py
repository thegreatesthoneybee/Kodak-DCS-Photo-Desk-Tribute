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
SAMPLE = Path(os.environ.get("PHOTODESK_SAMPLE", ROOT / "raw/ProBack645/4H6B9130.DCR"))
SLR_C_SAMPLE = ROOT / "raw/SLRc/F5KG7096.DCR"
PRO14N_SAMPLE = ROOT / "raw/14n/CX458821.DCR"

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


def test_camera_looks():
    looks = [("P.icm", "Kodak DCS Portrait Look"), ("PG.icm", "Kodak DCS Portrait Grayscale Look"),
             ("PR.icm", "Kodak DCS Portrait Reduced Look"), ("W.icm", "Kodak DCS Wedding Look")]
    assert [n for n, _ in camera.capabilities("DCS660M").looks(looks)] == ["P.icm", "PG.icm"]
    assert [n for n, _ in camera.capabilities("DCS620x").looks(looks)] == ["P.icm", "PG.icm", "PR.icm"]
    assert camera.capabilities("DCS Pro SLR/n").looks(looks) == looks
    assert camera.capabilities("DCS315").noise_types == ("none", "basic")
    assert camera.capabilities("DCS Pro Back").balances == ("preset", "click")
    assert camera.capabilities("ProBack645").balances == ("preset", "click", "custom", "auto")
    assert not camera.capabilities("ProBack645").white_point
    assert camera.capabilities("DCS Pro 14N").white_point
    assert camera.capabilities("DCS Pro SLR/n").white_point
    assert camera.capabilities("DCS560").noise_types == ("none", "basic", "advanced")
    assert camera.capabilities("DCS760C").noise_types == ("none", "basic", "advanced", "moire")
    assert camera.capabilities("ProBack645").noise_types == ("none", "basic", "advanced", "moire")
    assert camera.capabilities("DCS Pro SLR/n").noise_types == ("none", "basic", "advanced", "moire", "expert")


def test_custom_looks_in_system_folders(profile_dir, tmp_path, monkeypatch):
    import shutil
    shutil.copy(profile_dir / "DCSSepia1.icm", tmp_path / "MyCustomLook.icm")
    shutil.copy(profile_dir / "KODAK_sRGB.icm", tmp_path / "monitor.icm")
    monkeypatch.setattr(profiles, "system_color_dirs_profiles", lambda: iter(sorted(tmp_path.iterdir())))
    names = [n for n, _ in pipeline.available_looks(profile_dir)]
    assert "MyCustomLook.icm" in names and "monitor.icm" not in names and "DCSProductLook.icm" in names
    assert pipeline.look_path("MyCustomLook.icm", profile_dir) == tmp_path / "MyCustomLook.icm"


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
    assert pb.exposure_range == 1.0 and "custom" in pb.balances and "basic" in pb.noise_types
    slr = camera.capabilities("DCS Pro SLR/n")
    assert slr.exposure_range == 2.0 and "expert" in slr.noise_types and slr.lens_optimization
    assert not camera.capabilities("DCS Pro 14N").lens_optimization


def test_unlock_all_camera_features():
    try:
        camera.set_unlock_all(True)
        c = camera.capabilities("DCS560C")
        assert c.lens_optimization and c.white_point and c.look_set is None
        assert c.noise_types == camera.ALL_NOISE_TYPES
        pb = camera.capabilities("DCS Pro Back")
        assert pb.balances == ("preset", "click", "custom", "auto") and pb.exposure_range == 2.0
        assert pb.auto_scene and pb.balance_file
        assert not camera.capabilities("Canon EOS 5D").unlocked     # only DCS raw files
    finally:
        camera.set_unlock_all(False)
    c = camera.capabilities("DCS560C")
    assert not c.lens_optimization and c.look_set and not camera.capabilities("ProBack645").auto_scene


def test_at_capture_defaults_per_camera():
    low, medium = pipeline.SHARPEN_PRESETS["low"], pipeline.SHARPEN_PRESETS["medium"]
    sharp = lambda s: (s.sharpen_amount, s.sharpen_radius, s.sharpen_threshold)
    assert sharp(pipeline.at_capture({"model": "DCS560C"})) == medium
    assert sharp(pipeline.at_capture({"model": "ProBack645"})) == low
    assert sharp(pipeline.at_capture({"model": "DCS Pro SLR/c"})) == low
    assert sharp(pipeline.at_capture({"model": "ProBack645", "sharpening": False})) == (0, 0, 0)
    assert pipeline.at_capture({"model": "DCS Pro SLR/c", "lens_optimization": 24}).lens_optimization == 24


@pytest.mark.skipif(not (SLR_C_SAMPLE.is_file() and PRO14N_SAMPLE.is_file()), reason="SLR samples not available")
def test_lens_strength_recorded_by_camera():
    # Kodak IFD tag 2654 = [56, 0, strength, 0] on Pro SLR/c and SLR/n files.
    assert camera.capture_info(SLR_C_SAMPLE, "DCS Pro SLR/c")["lens_optimization"] == 24
    assert pipeline.capture_settings(SLR_C_SAMPLE).lens_optimization == 24
    # The 14n has no Lens Optimization even though it writes the tag (as zeros).
    assert "lens_optimization" not in camera.capture_info(PRO14N_SAMPLE, "DCS Pro 14N")


DCS560_SAMPLE = ROOT / "raw/560/3R7D2667.TIF"


@pytest.mark.skipif(not (DCS560_SAMPLE.is_file() and SAMPLE.is_file()), reason="samples not available")
def test_custom_balance_uses_camera_balance_point():
    # Kodak tags 2120-2125 are the camera's balance points; Custom is tag 2124 (0x84C).
    bp = camera.balance_points(DCS560_SAMPLE)
    assert bp["custom"] == (288, 412, 130) and bp["Daylight"] == (2069, 2303, 1175)
    assert camera.capture_info(DCS560_SAMPLE, "DCS560C")["white_balance"] == "custom"
    raw560 = pipeline.decode(DCS560_SAMPLE, half_size=True)
    custom = pipeline.white_balance(raw560, pipeline.Settings(balance="custom"))
    np.testing.assert_allclose(custom, [412 / 288, 1, 412 / 130])
    # Older DCS images take the Preset balance from the lighting's balance point as well.
    np.testing.assert_allclose(pipeline.preset_wb(raw560, "Tungsten"), [2233 / 2310, 1, 2233 / 923])
    # The camera shot it with its Custom setting: At Capture selects the Custom balance.
    assert pipeline.capture_settings(DCS560_SAMPLE).balance == "custom"
    # Pro Back: Custom differs from Preset (it used to fall back to the as-shot balance).
    pb = pipeline.decode(SAMPLE, half_size=True)
    assert not np.allclose(pipeline.white_balance(pb, pipeline.Settings(balance="custom")),
                           pipeline.white_balance(pb, pipeline.Settings()))


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample not available")
def test_white_point_follows_balance_method():
    raw = pipeline.reduced(pipeline.decode(SAMPLE, half_size=True), 256)
    s = pipeline.Settings()
    # Preset balances map back to their lighting's temperature.
    assert pipeline.balance_temperature(raw, s) == pipeline.LIGHTING_CCT["Daylight"]
    assert pipeline.balance_temperature(raw, s.copy(lighting="Tungsten")) == pipeline.LIGHTING_CCT["Tungsten"]
    # The White Point slider starts at the method's temperature without changing the balance.
    t = pipeline.balance_temperature(raw, s.copy(balance="auto"))
    a = s.copy(balance="auto", white_point=t, white_point_ref=t)
    np.testing.assert_allclose(pipeline.white_balance(raw, a), pipeline.white_balance(raw, s.copy(balance="auto")))
    # Moving it warms/cools from there.
    assert not np.allclose(pipeline.white_balance(raw, a.copy(white_point=t + 1000)), pipeline.white_balance(raw, a))


def test_lens_auto_measures_corner_cast():
    import cv2
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - (w - 1) / 2) ** 2 / ((w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2 / ((h - 1) / 2) ** 2) / 2
    lab = np.zeros((h, w, 3), np.float32)
    lab[..., 0] = 40 + 30 * (xx / w)                 # neutral grey ramp ...
    lab[..., 1] = 12 * r2                            # ... with a magenta/yellow corner cast
    lab[..., 2] = 9 * r2
    img = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    k = pipeline.lens_auto(img)
    assert 80 <= k <= 99
    # the chosen strength removes the cast it measured
    rest = cv2.cvtColor(pipeline.lens_optimize(img, k), cv2.COLOR_RGB2Lab)
    corner = np.abs(rest[:10, :10, 1:]).mean()
    assert corner < 0.25 * np.abs(lab[:10, :10, 1:]).mean()
    # an even grey has no cast, a saturated image cannot be measured
    flat = np.full((h, w, 3), 0.4, np.float32)
    assert pipeline.lens_auto(flat) == 0
    red = np.zeros((h, w, 3), np.float32)
    red[..., 0] = 0.9
    assert pipeline.lens_auto(red) is None


def test_noise_presets_and_ranges():
    v = {"noise_type": "advanced", "noise_radius": 1, "noise_strength": 0, "noise_luma": 0}
    presets = pipeline.noise_presets()
    assert pipeline.apply_noise_preset(dict(v), "low", presets)["noise_radius"] == 2
    assert pipeline.apply_noise_preset(dict(v), "high", presets)["noise_radius"] == 3   # level 80 <= 83
    assert pipeline.apply_noise_preset(dict(v), "medium", presets)["noise_strength"] == 50
    moire = pipeline.apply_noise_preset(dict(v, noise_type="moire"), "high", presets)
    assert (moire["noise_radius"], moire["noise_strength"]) == (8, 80)
    assert pipeline.apply_noise_preset(dict(v, noise_type="basic"), "medium", presets)["noise_radius"] == 4
    assert pipeline.apply_noise_preset(dict(v, noise_type="expert"), "low", presets)["noise_strength"] == 50
    pb, old = camera.capabilities("ProBack645"), camera.capabilities("DCS560C")
    assert pb.noise_controls("moire") == "radius" and pb.noise_controls("expert") == "strength"
    assert old.noise_controls("moire") == "threshold" and old.noise_controls("advanced") == "none"
    if SLR_C_SAMPLE.is_file():
        # Pro SLR files carry the preset tables (Kodak tags 3695 and 3683).
        p = pipeline.noise_presets(SLR_C_SAMPLE)
        assert p["radius"] == ((20, 23), (50, 50), (80, 80)) and p["expert"] == (50, 70, 80)


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


@need_sample
def test_white_balance_file(tmp_path):
    from photodesk import wbfile
    out = tmp_path / "STUDIO.wb"
    wbfile.write(out, SAMPLE, (2.0, 1.0, 1.5))
    data = out.read_bytes()
    assert data[:8] == b"MM\x00\x2a\x00\x00\x00\x08"
    assert struct.unpack_from(">H", data, 8)[0] == len(wbfile.IFD0_TAGS)
    ifd0, kodak = wbfile.read_source(out)
    src0, _ = wbfile.read_source(SAMPLE)
    assert ifd0[0x827D] == src0[0x827D]          # camera serial number: the camera checks it
    r, g, b = wbfile.read_neutral(out)
    assert abs(r * 2.0 - g) <= 2 and abs(b * 1.5 - g) <= 2


def _tiff_with_gps(path, lat, lon, alt):
    """Minimal big-endian TIFF whose IFD0 points at a GPS IFD (as DCS cameras write it)."""
    gps = [(0, 1, 4, b"\x02\x02\x00\x00"), (1, 2, 2, b"S\0\0\0"), (2, 5, 3, None),
           (3, 2, 2, b"W\0\0\0"), (4, 5, 3, None), (5, 1, 1, b"\0\0\0\0"), (6, 5, 1, None)]
    rats = {2: lat, 4: lon, 6: alt}
    gps_off = 8 + 2 + 12 + 4
    data_off = gps_off + 2 + 12 * len(gps) + 4
    body, blobs = b"", b""
    for tag, typ, cnt, val in gps:
        if val is None:
            val = struct.pack(">I", data_off + len(blobs))
            blobs += b"".join(struct.pack(">II", n, d) for n, d in rats[tag])
        body += struct.pack(">HHI", tag, typ, cnt) + val
    head = b"MM\0*" + struct.pack(">I", 8) + struct.pack(">H", 1) + struct.pack(">HHII", 0x8825, 4, 1, gps_off)
    path.write_bytes(head + b"\0" * 4 + struct.pack(">H", len(gps)) + body + b"\0" * 4 + blobs)


def test_gps(tmp_path):
    f = tmp_path / "g.tif"
    _tiff_with_gps(f, [(33, 1), (52, 1), (300, 10)], [(151, 1), (12, 1), (360, 10)], [(58, 1)])
    gps = camera.read_gps(f)
    assert gps["lat"] == pytest.approx(-(33 + 52 / 60 + 30 / 3600))
    assert gps["lon"] == pytest.approx(-(151 + 12 / 60 + 36 / 3600))
    assert gps["alt"] == 58
    assert camera.gps_text(gps) == "Lat 33d 52' 30.0\" S  Lon 151d 12' 36.0\" W  Alt 58 m"
    img = np.full((100, 800, 3), 0.5, np.float32)
    out = export.render_gps(img, camera.gps_text(gps))
    assert out.dtype == np.float32 and out[:40, :400].max() == 1.0 and out[60:].max() == 0.5
    assert dict(export._exif({"gps": gps}, 300).get_ifd(0x8825))[1] == "S"


@need_sample
def test_gps_absent_in_sample():
    assert camera.read_gps(SAMPLE) is None       # the ProBack writes an empty GPS IFD


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


def test_system_language_uses_macos_preferred_language(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("PHOTODESK_LANG", raising=False)
    monkeypatch.setattr(ui_strings.sys, "platform", "darwin")
    monkeypatch.setattr(ui_strings.locale, "getlocale", lambda: ("en_US", "UTF-8"))
    monkeypatch.setattr(
        ui_strings.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='(\n    "ko-KR"\n)\n'),
    )
    assert ui_strings.system_language() == "Korean"


def test_gui_task_skips_stale_work_and_results():
    from photodesk.gui.views import Task

    called = []
    stale = [True]
    task = Task(None, lambda: called.append("work"), stale=lambda: stale[0])
    task.run()
    assert called == []

    def finish_work():
        called.append("work")
        stale[0] = True
        return "result"

    stale[0] = False
    task = Task(None, finish_work, stale=lambda: stale[0])
    task.signals.done.connect(lambda *_: called.append("result"))
    task.run()
    assert called == ["work"]


def test_histogram_overlays_channels_with_transparency():
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from photodesk.gui.views import HistogramWidget

    app = QApplication.instance() or QApplication([])
    widget = HistogramWidget()
    widget.resize(258, 100)
    widget.channels = {"r": True, "g": True, "b": True}
    widget.set_image(np.full((2, 2, 3), 100, dtype=np.uint8))
    image = QImage(widget.size(), QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    widget.render(image)
    overlap = image.pixelColor(101, 50)
    assert overlap.red() > 0 and overlap.green() > 0 and overlap.blue() > 0


def test_windows_app_icon_contains_high_resolution_variants():
    from PIL import Image

    icon = Image.open(ROOT / "packaging/PhotoDesk.ico")
    assert (256, 256) in {(w, h) for w, h, *_ in icon.info["sizes"]}


def test_settings_migrate_from_kodak_scope(tmp_path):
    from PySide6.QtCore import QSettings
    from photodesk.gui import app as gui_app
    old = QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat)   # never the real registry
    old.setValue("Appearance", "dark")
    new = QSettings(str(tmp_path / "new.ini"), QSettings.IniFormat)
    gui_app._migrate_settings(new, old)
    assert new.value("Appearance") == "dark"
    assert not old.allKeys()
    old.setValue("Appearance", "dark")
    new.setValue("Appearance", "light")
    gui_app._migrate_settings(new, old)         # existing settings are never overwritten
    assert new.value("Appearance") == "light"


def test_dark_appearance_recolors_icons_and_palette():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QIcon, QImage, QPalette, QPixmap
    from PySide6.QtWidgets import QApplication
    from photodesk.gui.app import _dark_icon, apply_appearance

    app = QApplication.instance() or QApplication([])
    apply_appearance(app, "light")
    light = app.palette().color(QPalette.ColorRole.Window)
    apply_appearance(app, "dark")
    dark = app.palette().color(QPalette.ColorRole.Window)
    assert dark.lightness() < light.lightness()

    pixels = QImage(16, 16, QImage.Format.Format_ARGB32)
    pixels.fill(Qt.transparent)
    pixels.setPixelColor(8, 8, QColor(20, 20, 20))
    dark_pixel = _dark_icon(QIcon(QPixmap.fromImage(pixels))).pixmap(16, 16).toImage().pixelColor(8, 8)
    assert dark_pixel.red() > 150
    apply_appearance(app, "light")


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
def test_interface_pack_replaces_original_exe(profile_dir, tmp_path, monkeypatch):
    from photodesk import peres
    from photodesk.gui import res as R
    exe = (profiles.ui_dir(profile_dir) / "PhotoDsk.exe").read_bytes()
    original = PEResources(exe)
    pack = peres.write_pack(original)
    assert len(pack) < len(exe) // 3 and b"This program cannot be run" not in pack
    packed = peres.load(pack)
    assert isinstance(packed, peres.PackResources)
    assert packed.strings() == original.strings() and packed.menu(128) == original.menu(128)
    assert packed.dialog(103).size == original.dialog(103).size and packed.icon(128) == original.icon(128)
    assert packed.bitmap(193) == original.bitmap(193) and packed.cursor(packed.ids(12)[0]) == original.cursor(original.ids(12)[0])
    # A release layout: profiles + translation table + pack, no PhotoDsk.exe and no help.
    release = tmp_path / "profiles"
    (release / "ui").mkdir(parents=True)
    (release / "ui" / "PhotoDeskUIStrings.u16").write_bytes((profiles.ui_dir(profile_dir) / "PhotoDeskUIStrings.u16").read_bytes())
    (release / "ui" / profiles.UI_PACK).write_bytes(pack)
    monkeypatch.setenv("PHOTODESK_PROFILES", str(release))
    res = R.Resources.load("Korean")
    assert isinstance(res.pe, peres.PackResources) and res.s(59148)


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
    fin_tif = export.process_file(jpg, tmp_path / "f.tif", "tiff", export.OutputOptions(tiff_16bit=True),
                                  pipeline.Settings(), profile_dir)
    assert tifffile.imread(fin_tif).dtype == np.uint8          # 16-bit does not apply to 8-bit sources


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
    src = pipeline.Settings(look="DCSPortraitLook.icm", crop=(0, 0, 0.5, 0.5))
    sheet.paste([b], src, ("look", "crop"))
    assert b.settings.crop == (0, 0, 0.5, 0.5) and b.settings.look != src.look   # finished: crop only
    assert sheet.save([a]) == [] and not pipeline.sidecar_path(a.path).is_file()   # finished file: not saved
    pipeline.save_settings(a.path, a.settings)
    sheet.rename([a, b], ["x01.jpg", "x02.jpg"])
    assert {e.path.name for e in sheet.entries} == {"x01.jpg", "x02.jpg"}
    assert pipeline.sidecar_path(tmp_path / "x01.jpg").is_file()
    sheet.undo()
    assert {e.path.name for e in sheet.entries} == {"a.jpg", "b.jpg"}
    # An existing name is skipped; on_error decides whether to rename the rest.
    Image.new("RGB", (8, 8)).save(tmp_path / "y01.jpg")
    errors = []
    done = sheet.rename([a, b], ["y01.jpg", "y02.jpg"], lambda e, kind: errors.append((e.path.name, kind)) or True)
    assert errors == [("a.jpg", "exists")] and done == [b] and b.path.name == "y02.jpg"
    sheet.undo()
    assert b.path.name == "b.jpg"
    (tmp_path / "y01.jpg").unlink()
    gone = sheet.delete([b], lambda p: p.unlink() or True)
    assert gone == [b] and not (tmp_path / "b.jpg").exists()


@need_sample
def test_paste_click_balance_same_camera_only(tmp_path):
    import shutil
    from photodesk.gui.model import Sheet
    shutil.copy(SAMPLE, tmp_path / "a.DCR")
    shutil.copy(SAMPLE, tmp_path / "b.DCR")
    sheet = Sheet(tmp_path)
    a, b = sheet.entries
    assert a.serial == "PB645M-00152"
    src = pipeline.Settings(balance="click", click_wb=(1.2, 1.0, 1.4), exposure=0.5)
    fields = ("balance", "click_wb", "exposure")
    sheet.paste([a], src, fields, source_serial="OTHER-1")
    assert a.settings.exposure == 0.5 and a.settings.balance != "click"
    sheet.paste([b], src, fields, source_serial=a.serial)
    assert b.settings.balance == "click" and b.settings.click_wb == (1.2, 1.0, 1.4)


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
