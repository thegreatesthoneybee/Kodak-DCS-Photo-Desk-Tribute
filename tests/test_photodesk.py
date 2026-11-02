"""Tests run against the original installer and sample raw when available.

Set PHOTODESK_INSTALLER / PHOTODESK_SAMPLE to override the default locations.
"""
import io
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from photodesk import camera, demosaic, export, icc, iptc, mscab, pipeline, profiles, ui_strings
from photodesk.peres import PEResources

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = Path(os.environ.get("PHOTODESK_INSTALLER", ROOT / "vendor/original/Photo_Desk_V_4_3_0.exe"))
SAMPLE = Path(os.environ.get("PHOTODESK_SAMPLE", ROOT / "raw/ProBack645/4H6B9130.DCR"))
SLR_C_SAMPLE = ROOT / "raw/SLRc/F5KG7096.DCR"
SLR_N_SAMPLE = ROOT / "raw/SLRn/GQ432034.DCR"
if os.environ.get("PHOTODESK_SLRN_SAMPLE"):     # e.g. a copy while Photo Desk holds the original open
    SLR_N_SAMPLE = Path(os.environ["PHOTODESK_SLRN_SAMPLE"])
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


def _tone_curve_raw():
    return pipeline.RawImage(Path("tone-curve-test.dcr"), "Kodak", "DCS660M",
                             np.full((2, 2, 3), 0.1, np.float32), np.ones(3), np.ones(3),
                             np.zeros((3, 3)), np.eye(3, dtype=np.float32), 0, {})


def _stub_tone_curve_renderer(monkeypatch):
    calls = []
    axis = np.linspace(0, 1, pipeline.LUT_SIZE, dtype=np.float32)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)

    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.ones(3, np.float32))
    monkeypatch.setattr(pipeline, "source_profile_for", lambda _raw, settings, _pdir:
                        f"{settings.lighting}-source.icm")
    monkeypatch.setattr(pipeline, "resolve_space", lambda space, _pdir: str(space))

    def color_lut(source, look, space, pdir):
        calls.append((source, look, space, pdir))
        scale = 0.6 if space == "wide-space.icm" else 1.0
        if look == "other-look.icm":
            scale *= 0.8
        if source == "Tungsten-source.icm":
            scale *= 0.7
        return identity * scale

    monkeypatch.setattr(pipeline, "color_lut", color_lut)
    return _tone_curve_raw(), calls


def test_tone_curve_uses_renderer_tone_and_selected_output_profile(monkeypatch, tmp_path):
    raw, calls = _stub_tone_curve_renderer(monkeypatch)
    settings = pipeline.Settings(look="selected-look.icm")
    ev, neutral = pipeline.tone_curve(raw, settings, "srgb-space.icm", tmp_path, n=64)
    _, adjusted = pipeline.tone_curve(raw, settings.copy(tone=(-9.5, -1.0, 3.4)),
                                      "srgb-space.icm", tmp_path, n=64)
    _, wide = pipeline.tone_curve(raw, settings, "wide-space.icm", tmp_path, n=64)
    _, other_look = pipeline.tone_curve(raw, settings.copy(look="other-look.icm"),
                                        "srgb-space.icm", tmp_path, n=64)
    _, other_source = pipeline.tone_curve(raw, settings.copy(lighting="Tungsten"),
                                          "srgb-space.icm", tmp_path, n=64)

    assert len(ev) == len(neutral) == 64 and np.isfinite(neutral).all()
    assert not np.array_equal(adjusted, neutral)
    assert not np.array_equal(wide, neutral)
    assert not np.array_equal(other_look, neutral)
    assert not np.array_equal(other_source, neutral)
    assert calls[0] == ("Daylight-source.icm", "selected-look.icm", "srgb-space.icm", str(tmp_path))


def test_tone_curve_ignores_controls_when_disabled(monkeypatch, tmp_path):
    raw, _ = _stub_tone_curve_renderer(monkeypatch)
    neutral = pipeline.Settings(tone_enabled=False)
    adjusted = pipeline.Settings(tone=(-9.5, -1.0, 3.4), tone_enabled=False)
    _, neutral_curve = pipeline.tone_curve(raw, neutral, "srgb-space.icm", tmp_path, n=32)
    _, disabled_curve = pipeline.tone_curve(raw, adjusted, "srgb-space.icm", tmp_path, n=32)
    _, enabled_curve = pipeline.tone_curve(raw, adjusted.copy(tone_enabled=True), "srgb-space.icm", tmp_path, n=32)
    np.testing.assert_array_equal(disabled_curve, neutral_curve)
    assert not np.array_equal(enabled_curve, neutral_curve)
    assert pipeline.tone_curve(None, adjusted, "srgb-space.icm", tmp_path) is None


def test_tone_curve_uses_output_rgb_weights(monkeypatch, tmp_path):
    raw, _ = _stub_tone_curve_renderer(monkeypatch)
    monkeypatch.setattr(icc, "trilinear", lambda lut, values, power: np.broadcast_to(
        np.array([1.0, 0.0, 0.0], np.float32), values.shape))
    _, values = pipeline.tone_curve(raw, pipeline.Settings(), "srgb-space.icm", tmp_path, n=4)
    np.testing.assert_allclose(values, 0.299)


@pytest.mark.parametrize("value,expected", [(1.25, 1.3), (-1.25, -1.2), (1.0, 1.0), (0.0, 0.0)])
def test_tone_marker_midpoints_round_toward_positive_infinity(value, expected):
    assert pipeline.tone_marker_value(value) == float(np.float32(expected))


def test_tone_marker_preserves_negative_zero():
    assert np.signbit(pipeline.tone_marker_value(-0.0))


@pytest.mark.parametrize("source,tone", [
    ("DCSProSLRnDaylight.icm", "DCSProSLRnToneCurve.icm"),
    ("DCSProSLRcDaylight.icm", "DCSProSLRcToneCurve.icm"),
    ("DCSPro14Daylight.icm", "DCSPro14ToneCurve.icm"),
])
def test_camera_color_chain_uses_tone_profile(profile_dir, monkeypatch, source, tone):
    seen = []
    original = pipeline._profile
    def track(path):
        seen.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(pipeline, "_profile", track)
    pipeline.color_lut.cache_clear()
    space = str(profile_dir / "sRGB Color Space Profile.icm")
    pipeline.color_lut(source, "DCSProductLook.icm", space, str(profile_dir))
    assert tone in seen
    pipeline.color_lut.cache_clear()


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


def _tiff_with_lens_baseline(path, tag_type=4, count=4, values=(56, 0, 10, 0)):
    """Write a minimal little-endian TIFF with a Kodak tag 0xa5e record."""
    if tag_type is None:
        root = (struct.pack("<H", 1) + struct.pack("<HHII", camera.KODAK_IFD, 4, 1, 26)
                + b"\0" * 4)
        path.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + b"\0" * 6)
        return
    payload = struct.pack("<" + ("H" if tag_type == 3 else "I") * count, *values[:count])
    kodak_ifd_offset = 26
    payload_offset = kodak_ifd_offset + 2 + 12 + 4
    field = payload if len(payload) <= 4 else struct.pack("<I", payload_offset)
    field = field.ljust(4, b"\0")
    root = (struct.pack("<H", 1) + struct.pack("<HHII", camera.KODAK_IFD, 4, 1, kodak_ifd_offset)
            + b"\0" * 4)
    kodak = (struct.pack("<H", 1) + struct.pack("<HHI", camera.LENS_BASELINE_TAG, tag_type, count)
             + field + b"\0" * 4)
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + kodak
                     + (payload if len(payload) > 4 else b""))


def _omen_record(**changes):
    grid = np.full((48, 71), 4096, np.int64)
    grid[:, :35] = 3584                              # left half: blue gain map below unity
    record = {"grid": grid, "unit": 0x1000, "scale": (0.0, 2.5), "baseline": 10}
    record.update(changes)
    return record


def test_omen_tiles_are_unity_at_the_camera_strength_and_scale_blue_otherwise():
    record = _omen_record()
    same = demosaic.omen_tiles(record["grid"], 0x1000, (0.0, 2.5), 10, 10)
    assert same.shape == demosaic.OMEN_TILES and np.all(same == 4096)
    # x = -512, A = 0, B = 0.25: 4096 * 4096 / (4096 - 128) -> 4228 (truncated), unit gain at x = 0
    off = demosaic.omen_tiles(record["grid"], 0x1000, (0.0, 2.5), 0, 10)
    assert off[0, 0] == 4228 and off[0, 280] == 4096 and off[188, 282] == 4096
    # nodes on every fourth tile, (a + b + 1) >> 1 midpoints in between
    assert off[0, 136] == 4228 and off[0, 140] == 4096
    assert list(off[0, 137:140]) == [4195, 4162, 4129]
    assert np.all(off[1:4, 137] == off[0, 137])


def test_apply_omen_cfa_scales_only_blue_sites():
    cfa = np.full((4, 4), 1000, np.int64)
    tiles = np.full(demosaic.OMEN_TILES, 4096 + 2048, np.int64)
    out = demosaic.apply_omen_cfa(cfa, tiles)
    assert np.all(out[1::2, 0::2] == 1500)
    assert np.all(out[0::2, :] == 1000) and np.all(out[1::2, 1::2] == 1000)
    assert demosaic.apply_omen_cfa(np.full((2, 2), 4000), tiles)[1, 0] == 4095


def test_lens_omen_gain_follows_strength_tiles_and_region_origin():
    from types import SimpleNamespace

    raw = SimpleNamespace(lens_omen=_omen_record(), scale=1.0, origin=(0, 0), white_level=3700)
    assert pipeline.lens_omen_gain(raw, 10, (32, 32)) is None
    assert pipeline.lens_omen_gain(raw, 100, (32, 32)) is None
    gain = pipeline.lens_omen_gain(raw, 0, (32, 32))
    np.testing.assert_allclose(gain[0, 0], 4228 / 4096)
    shifted = SimpleNamespace(**{**vars(raw), "origin": (0, 2240)})
    np.testing.assert_allclose(pipeline.lens_omen_gain(shifted, 0, (32, 32)), 1.0)
    half = SimpleNamespace(**{**vars(raw), "scale": 0.5})
    np.testing.assert_allclose(pipeline.lens_omen_gain(half, 0, (16, 16)), gain[::2, ::2])


def test_lens_omen_reads_the_slrn_record():
    if not SLR_N_SAMPLE.is_file():
        pytest.skip("SLR/n sample not available")
    record = camera.lens_omen(SLR_N_SAMPLE, "DCS Pro SLR/n")
    assert record["grid"].shape == (48, 71) and record["unit"] == 4096
    assert record["scale"] == (0.0, 2.5) and record["baseline"] == 10
    assert record["grid"][24, 35] == 4103
    assert camera.lens_omen(SLR_N_SAMPLE, "DCS Pro 14N") is None


DCS560_SAMPLE = ROOT / "raw/560/3R7D2667.TIF"


def test_stored_process_levels_reject_invalid_blocks(tmp_path, monkeypatch):
    import struct
    path = tmp_path / "sample.dcr"
    path.write_bytes(b"MM")
    data = bytearray(112)
    struct.pack_into(">II", data, 4, 2, 112)
    for offset, pair in zip((68, 76, 84), ((-19, 2), (-1, 5), (4, 1))):
        struct.pack_into(">ii", data, offset, *pair)
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *args: {5011: tuple(data)})
    assert camera.stored_process_levels(path) == (-9.5, -0.2, 4.0)
    assert camera.stored_process_parameters(path)["tone_enabled"] is False
    struct.pack_into(">I", data, 92, 1)
    assert camera.stored_process_parameters(path)["tone_enabled"] is True
    struct.pack_into(">i", data, 72, 0)
    assert camera.stored_process_levels(path) is None
    struct.pack_into(">I", data, 4, 3)
    assert camera.stored_process_levels(path) is None
    struct.pack_into(">i", data, 72, 2)
    assert camera.stored_process_levels(path) == (-9.5, -0.2, 4.0)


def test_stored_process_levels_little_endian_and_sidecar(tmp_path, monkeypatch):
    import struct
    path = tmp_path / "sample.dcr"
    path.write_bytes(b"II")
    data = bytearray(112)
    struct.pack_into("<II", data, 4, 2, 112)
    for offset, pair in zip((68, 76, 84), ((-19, 2), (-1, 5), (4, 1))):
        struct.pack_into("<ii", data, offset, *pair)
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *args: {5011: tuple(data)})
    assert camera.stored_process_levels(path) == (-9.5, -0.2, 4.0)
    saved = pipeline.Settings(tone=(-8.0, 0.0, 3.0))
    pipeline.save_settings(path, saved)
    assert pipeline.load_settings(path).tone == saved.tone


def test_saved_kelvin_tuple_normalizes_preset_illumination(tmp_path, monkeypatch):
    import struct
    path = tmp_path / "sample.dcr"
    path.write_bytes(b"II")
    data = bytearray(112)
    struct.pack_into("<II", data, 4, 2, 112)
    struct.pack_into("<III", data, 16, 3200, 1, 3)
    struct.pack_into("<3h", data, 60, -100, 4, 100)
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *args: {5011: tuple(data)})
    params = camera.stored_process_parameters(path)
    assert (params["white_point"], params["kelvin_mode"], params["kelvin_illumination"]) == (3200, 1, "Tungsten")
    assert params["color_balance"] == (-100, 4, 100)
    monkeypatch.setattr(camera, "read_make_model", lambda *_: ("Kodak", "DCS Pro SLR/n"))
    monkeypatch.setattr(camera, "capture_info", lambda *_: {"model": "DCS Pro SLR/n"})
    monkeypatch.setattr(camera, "balance_curves", lambda *_: {})
    assert pipeline.load_settings(path).color_balance == (-100, 4, 100)
    struct.pack_into("<h", data, 60, -101)
    assert camera.stored_process_parameters(path)["color_balance"] == (0, 0, 0)
    struct.pack_into("<III", data, 16, 5200, 5, 2)
    params = camera.stored_process_parameters(path)
    assert (params["kelvin_mode"], params["kelvin_illumination"]) == (5, "Fluorescent")
    struct.pack_into("<I", data, 20, 7)
    assert "kelvin_mode" not in camera.stored_process_parameters(path)


def test_stored_process_levels_from_slrn_sample(tmp_path):
    sample = SLR_N_SAMPLE
    if not sample.is_file():
        pytest.skip("SLR/n sample not available")
    # A copy without the user's sidecar, so only the camera-recorded settings are read.
    path = tmp_path / sample.name
    shutil.copyfile(sample, path)
    np.testing.assert_allclose(camera.stored_process_levels(path), (-9.5, -0.2, 4.0), atol=1e-5)
    settings = pipeline.load_settings(path)
    np.testing.assert_allclose(settings.tone, (-9.5, -0.2, 4.0), atol=1e-5)
    assert settings.white_point == 5000
    assert settings.kelvin_mode == 0 and settings.kelvin_illumination == "Daylight"
    raw = pipeline.decode(path, half_size=True)
    np.testing.assert_allclose(raw.balance_factors["Daylight"], (1.0, 0.965, 0.98), atol=1e-6)
    assert settings.saved_balance_point == (2048, 2532, 1771)
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [2532 / 2048, 1.0, 2532 / 1771])
    assert not np.allclose(pipeline.white_balance(raw, settings),
                           pipeline.white_balance(raw, settings.copy(white_point=5500)))
    for lighting, expected in (("Daylight", 5500), ("Tungsten", 3200),
                               ("Fluorescent", 5000), ("Flash", 5500)):
        assert pipeline.balance_temperature(raw, settings.copy(lighting=lighting)) == expected


def test_kelvin_point_quantizes_and_clamps(monkeypatch):
    from types import SimpleNamespace
    curve = [0.0] * 27
    curve[0], curve[4], curve[8] = -50.0, 500.49, 20000.0
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), balance_curves={"Daylight": curve})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.ones(3))
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings()), [100.0, 20.0, 1.0])
    raw.balance_factors = {"Daylight": (1.0, 0.965, 0.98)}
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings()), [98.0, 9800 / 482, 1.0])


def test_kelvin_forward_binary32_spills_straddle_half_rounding():
    curve = [0.0] * 27
    for offset in (0, 4, 8):
        curve[offset] = 100.0
        curve[offset + 1] = 0.01
    # At 5000 K, binary32 0.01 * 50 is just below 0.5. Red/green spill to
    # binary32 (100.5); blue stays extended and truncates 100.999... to 100.
    np.testing.assert_array_equal(pipeline.kelvin_point(curve, 5000), [101.0, 101.0, 100.0])


def test_kelvin_forward_temperature_is_an_unsigned_word():
    curve = [0.0] * 27
    for offset, base, slope in zip((0, 4, 8), (100, 200, 300), (1, 2, 3)):
        curve[offset], curve[offset + 1] = base, slope
    np.testing.assert_array_equal(pipeline.kelvin_point(curve, 67536),
                                  pipeline.kelvin_point(curve, 2000))
    assert pipeline.kelvin_point(curve, float("nan")) is None
    assert pipeline.kelvin_point(curve, float("inf")) is None


@pytest.mark.parametrize(("factor", "rounded_word"), [
    (0.50004995, 5000),
    (0.50004999, 5001),
])
def test_kelvin_scale_uses_binary32_factors_around_half(monkeypatch, factor, rounded_word):
    from types import SimpleNamespace
    curve = [0.0] * 27
    curve[0] = curve[4] = curve[8] = 10000.0
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), balance_curves={"Daylight": curve},
                          balance_factors={"Daylight": (1.0, factor, 1.0)})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.ones(3))
    native_product = 10000 * float(np.float32(factor))
    assert (native_product < 5000.5) == (rounded_word == 5000)
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings()),
                               [1.0, 10000 / rounded_word, 1.0])


def test_kelvin_scale_stores_words_and_rejects_zero_channels(monkeypatch):
    from types import SimpleNamespace
    curve = [0.0] * 27
    curve[0] = curve[4] = curve[8] = 10000.0
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), balance_curves={"Daylight": curve},
                          balance_factors={"Daylight": (1.0, 6.5537, 1.0)})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.ones(3))
    # 65537 is truncated to a WORD, so the green channel wraps to 1 before normalization.
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings()), [1.0, 10000.0, 1.0])
    raw.balance_factors["Daylight"] = (1.0, 6.5536, 1.0)
    # This product stores as WORD 0; do not convert the native invalid channel into inf/NaN.
    with pytest.raises(ValueError, match="zero channel"):
        pipeline.white_balance(raw, pipeline.Settings())


def test_saved_kelvin_uses_its_own_illumination(monkeypatch):
    from types import SimpleNamespace
    daylight, tungsten = [0.0] * 27, [0.0] * 27
    daylight[0], daylight[4], daylight[8] = 1000, 1000, 1000
    tungsten[0], tungsten[4], tungsten[8] = 1000, 2000, 3000
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True),
                          balance_curves={"Daylight": daylight, "Tungsten": tungsten},
                          balance_factors={"Tungsten": (1.0, 0.5, 1.0)})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.ones(3))
    settings = pipeline.Settings(lighting="Daylight", kelvin_mode=1, kelvin_illumination="Tungsten")
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [3.0, 3.0, 1.0])


@pytest.mark.parametrize("mode", range(4))
def test_saved_basic_kelvin_modes_override_nonpreset_balance(monkeypatch, mode):
    from types import SimpleNamespace
    illuminations = camera.ILLUMINANTS
    curves, balance_factors = {}, {}
    bases = {}
    slopes = {}
    for index, illumination in enumerate(illuminations):
        base = np.array([900, 1700, 2800], dtype=np.float64) + index * np.array([125, 175, 225])
        slope = np.array([3, 1, -2], dtype=np.float64) + index * np.array([0.5, -0.25, 0.2])
        curve = [0.0] * 27
        for channel, offset in enumerate((0, 4, 8)):
            curve[offset] = base[channel]
            curve[offset + 1] = slope[channel]
        factor = np.array([1.0 + 0.05 * index, 0.8 + 0.1 * index, 1.2 - 0.05 * index])
        curves[illumination] = curve
        balance_factors[illumination] = factor
        bases[illumination] = base
        slopes[illumination] = slope

    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), balance_curves=curves,
                          balance_factors=balance_factors)
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.array([9.0, 8.0, 7.0]))
    illumination = illuminations[mode]
    settings = pipeline.Settings(balance="click", lighting=illuminations[(mode + 1) % 4],
                                 white_point=5000, white_point_ref=5000, kelvin_mode=mode,
                                 kelvin_illumination=illumination,
                                 saved_balance_point=(4000, 2000, 1000))

    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [1.0, 2.0, 4.0])
    edited = settings.copy(white_point=5200)
    assert edited.saved_balance_point is None
    assert edited.kelvin_mode == mode
    point = np.floor(bases[illumination] + slopes[illumination] * 52 + 0.5)
    point = np.floor(point * balance_factors[illumination] + 0.5)
    expected = np.max(point) / point
    np.testing.assert_allclose(pipeline.white_balance(raw, edited), expected)


def test_nonpreset_balance_without_saved_kelvin_keeps_balance_method(monkeypatch):
    from types import SimpleNamespace
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True),
                          balance_curves={"Daylight": [1000.0] * 27},
                          balance_factors={"Daylight": (0.5, 0.5, 0.5)})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.array([1.7, 1.0, 2.3]))
    settings = pipeline.Settings(balance="click", white_point=5000, white_point_ref=5000)
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [1.7, 1.0, 2.3])


def test_invalid_preset_factor_triplet_is_ignored(monkeypatch):
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *_: {2130: (1.0, 0.099, 1.0), 2131: (1.0, 0.1, 1.0)})
    factors = camera.balance_factors("sample.dcr")
    assert factors["Daylight"] == (1.0, 1.0, 1.0)
    assert factors["Tungsten"] == (1.0, 0.1, 1.0)


def test_saved_auto_kelvin_uses_capture_calibration(tmp_path, monkeypatch):
    import shutil
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("SLR/c sample not available")
    path = tmp_path / SLR_C_SAMPLE.name
    shutil.copy2(SLR_C_SAMPLE, path)
    settings = pipeline.load_settings(path)
    assert (settings.kelvin_mode, settings.white_point, settings.white_point_ref) == (5, 5200, 5200)
    assert settings.saved_balance_point == (2048, 2621, 1934)
    assert settings.exposure == 0.5
    raw = pipeline.decode(path, half_size=True)
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.array([1.2, 1.0, 1.3]))
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [2621 / 2048, 1.0, 2621 / 1934])
    temperature, factors = pipeline.auto_kelvin_calibration(raw)
    assert temperature == 4818
    np.testing.assert_allclose(factors, [1, 0.98190867, 1.03181514], atol=1e-6)
    np.testing.assert_allclose(pipeline.white_balance(raw, settings.copy(white_point=5000)),
                               [2577 / 2048, 1.0, 2577 / 1864])
    assert settings.copy(white_point=5300).saved_balance_point is None
    assert settings.copy(balance="auto").saved_balance_point is None
    assert settings.copy(lighting="Tungsten").saved_balance_point is None
    assert settings.copy(click_wb=(1.1, 1.0, 1.2)).saved_balance_point is None
    assert settings.copy(balance="auto").kelvin_mode is None
    assert settings.copy(lighting="Tungsten").kelvin_mode == 5
    assert settings.copy(lighting="Tungsten").kelvin_illumination == "Tungsten"
    assert settings.copy(white_point=5000).kelvin_mode == 5
    assert settings.copy(exposure=0).saved_balance_point == settings.saved_balance_point
    pipeline.save_settings(path, settings)
    assert pipeline.load_settings(path).kelvin_mode == 5


def test_auto_kelvin_calibration_requires_valid_metadata(monkeypatch):
    tags = {1019: (0,), 2121: (2048, 1956, 1050), 2122: (2048, 2631, 1646),
            2125: (4695, 5872, 4265), 2011: (1,) * 9, 2012: (1,) * 9,
            3110: (2571,), 3122: (2871,), 3125: (1,)}
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *args: tags)
    assert camera.auto_kelvin_illumination("sample.dcr") == "Daylight"
    assert camera.auto_capture_settings("sample.dcr") == ("Daylight", -1.0)
    tags[3110] = (3171,)
    assert camera.auto_capture_settings("sample.dcr") == ("Daylight", 1.0)
    tags[3110] = (2961,)
    assert camera.auto_capture_settings("sample.dcr")[1] == float(np.float32(0.3))
    tags[2011] = (0,) * 9
    assert camera.auto_capture_settings("sample.dcr") == (None, None)
    tags[2011] = (1,) * 9
    tags[2121] = (2048, 1, 1050)
    assert camera.auto_capture_settings("sample.dcr") == (None, None)
    tags[2121] = (2048, 1956, 1050)
    tags[3125] = (0,)
    assert camera.auto_kelvin_illumination("sample.dcr") is None
    tags[3125] = (1,)
    tags[2125] = (4095, 4095, 4095)
    assert camera.auto_kelvin_illumination("sample.dcr") is None


@pytest.mark.parametrize("path,illumination,temperature,factors,point", [
    (ROOT / "raw/SLRc/F5KG7096.DCR", "Fluorescent", 6174, (1, 0.98863626, 0.87797266), (2048, 2579, 1529)),
    (SLR_N_SAMPLE, "Daylight", 5782, (1, 0.97900057, 0.94161063), (2048, 2764, 1994)),
])
def test_custom_kelvin_calibration_from_capture(path, illumination, temperature, factors, point):
    if not path.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(path, half_size=True)
    assert raw.custom_kelvin_illumination == illumination
    actual_temperature, actual_factors = pipeline.custom_kelvin_calibration(raw)
    assert actual_temperature == temperature
    np.testing.assert_allclose(actual_factors, factors, atol=1e-6)
    settings = pipeline.Settings(balance="custom", white_point=6000, kelvin_mode=4,
                                 kelvin_illumination=illumination)
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), max(point) / np.asarray(point))


def test_kelvin_mode_six_recomputes_factors():
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    settings = pipeline.Settings(white_point=5000, kelvin_mode=6, kelvin_illumination="Daylight")
    np.testing.assert_allclose(pipeline.white_balance(raw, settings), [2526 / 2048, 1.0, 2526 / 1664])
    assert not np.allclose(pipeline.white_balance(raw, settings.copy(white_point=5100)),
                           pipeline.white_balance(raw, settings))


@pytest.mark.parametrize("inverse,temperature", [
    ((0, 1000, 4000), 2500), ((2000, 0, 4000), 3000), ((2000, 4000, 0), 3000),
])
def test_kelvin_inverse_selects_coefficient_branches(inverse, temperature):
    from types import SimpleNamespace
    curve = [0.0] * 27
    curve[0], curve[4], curve[8] = 2048, 2000, 1800
    curve[12], curve[16], curve[20] = inverse
    raw = SimpleNamespace(balance_curves={"Daylight": curve}, balance_factors={"Daylight": (1, 1, 1)})
    assert pipeline._kelvin_calibration(raw, "Daylight", (2048, 2048, 2048), 5)[0] == temperature
    assert pipeline._kelvin_calibration(raw, "Daylight", (2048, 2048, 2048), 4) is None
    assert pipeline._kelvin_calibration(raw, "Daylight", (1, 1, 1), 5) is None


def test_stored_render_record_checks_size(tmp_path, monkeypatch):
    import struct
    path = tmp_path / "sample.dcr"
    path.write_bytes(b"II")
    data = bytearray(72)
    struct.pack_into("<II", data, 4, 1, 72)
    struct.pack_into("<i", data, 32, -500000)
    struct.pack_into("<I", data, 36, 1)
    struct.pack_into("<3H", data, 40, 2048, 2500, 1800)
    struct.pack_into("<I", data, 0, sum(data[4:]))
    monkeypatch.setattr(camera, "read_kodak_tags", lambda *args: {1021: tuple(data)})
    assert camera.stored_render_parameters(path) == {"exposure": -0.5, "lighting": "Tungsten", "balance_point": (2048, 2500, 1800)}
    data[0] ^= 1  # The original accumulates the checksum but does not reject on mismatch.
    assert camera.stored_render_parameters(path)["exposure"] == -0.5
    struct.pack_into("<I", data, 4, 7)
    assert camera.stored_render_parameters(path)["exposure"] == -0.5
    struct.pack_into("<I", data, 8, 73)
    assert camera.stored_render_parameters(path) == {}
    struct.pack_into("<I", data, 8, 36)
    del data[36:]
    assert camera.stored_render_parameters(path) == {"exposure": -0.5, "lighting": "Daylight"}


def test_long_stored_records_read_only_supported_prefix(tmp_path):
    import struct
    path = tmp_path / "sample.dcr"
    root = struct.pack("<H", 1) + struct.pack("<HHII", 0x8290, 4, 1, 26) + b"\0" * 4
    kodak = (struct.pack("<H", 2) + struct.pack("<HHII", 1021, 7, 128, 56)
             + struct.pack("<HHII", 5011, 7, 136, 184) + b"\0" * 4)
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + kodak + bytes(range(128)) + bytes(range(136)))
    tags = camera.read_kodak_tags(path, (1021, 5011))
    assert tags[1021] == tuple(range(72))
    assert tags[5011] == tuple(range(112))


def test_invalid_camera_matrix_denominator_rejects_tag(tmp_path):
    import struct
    path = tmp_path / "sample.dcr"
    root = struct.pack("<H", 1) + struct.pack("<HHII", 0x8290, 4, 1, 26) + b"\0" * 4
    kodak = struct.pack("<H", 1) + struct.pack("<HHII", 2011, 10, 9, 44) + b"\0" * 4
    matrix = struct.pack("<ii", 1, 0) + struct.pack("<ii", 0, 1) * 8
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + root + kodak + matrix)
    assert 2011 not in camera.read_kodak_tags(path, (2011,))


def test_pro_slr_color_balance_uses_camera_constants(monkeypatch):
    from types import SimpleNamespace
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), balance_curves={})
    monkeypatch.setattr(pipeline, "method_wb", lambda *_: np.ones(3))
    values = np.array([40, -20, 10])
    adjustment = 10.0 ** (0.0030103 * values)
    expected = adjustment / (adjustment @ np.array([0.299, 0.587, 0.114]))
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings(color_balance=tuple(values))), expected)
    np.testing.assert_allclose(pipeline.white_balance(raw, pipeline.Settings(color_balance=(101, 0, 0))), np.ones(3))


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


def test_finish_leaves_native_lens_optimization_to_scene_linear(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "lens_optimize", lambda img, strength: calls.append(strength) or img)
    caps = camera.capabilities("DCS Pro SLR/n")
    img = np.zeros((2, 2, 3), np.float32)

    def finish(strength, native):
        settings = pipeline.Settings(lens_optimization=strength, sharpen_amount=0)
        return pipeline.finish(img, settings, caps, native_lens=native)

    finish(10, True)     # the camera's record is applied before the colour chain
    finish(11, False)    # without a record the Lab heuristic remains
    finish(0, False)
    assert calls == [11]


def test_render_marks_native_lens_for_finish(monkeypatch):
    from types import SimpleNamespace

    raw = SimpleNamespace(caps=camera.capabilities("DCS Pro SLR/n"), scale=0.5, flip=0,
                          lens_omen=_omen_record(), xyz_from_cam=np.eye(3, dtype=np.float32))
    monkeypatch.setattr(pipeline, "source_profile_for", lambda *_: "source.icm")
    monkeypatch.setattr(pipeline, "color_lut", lambda *args: None)
    monkeypatch.setattr(pipeline, "resolve_space", lambda *_: "space.icm")
    monkeypatch.setattr(pipeline, "pro_slr_erimm", lambda *_: None)
    monkeypatch.setattr(pipeline, "scene_linear", lambda *_, **__: np.zeros((1, 1, 3), np.float32))
    monkeypatch.setattr(pipeline.icc, "trilinear", lambda *_args, **_kwargs: np.zeros((1, 1, 3), np.float32))
    seen = []

    def finish(img, settings, caps, scale, native_lens=False):
        seen.append((caps, scale, native_lens))
        return img

    monkeypatch.setattr(pipeline, "finish", finish)
    pipeline.render(raw, pipeline.Settings(sharpen_amount=0), profile_directory=".")
    assert seen == [(raw.caps, 0.5, True)]


def test_render_linear_uses_white_balanced_exposed_toned_scene_without_profiles(monkeypatch):
    raw = pipeline.RawImage(Path("linear-render-test.dcr"), "Kodak", "DCS660M",
                            np.array([[[0.04, 0.08, 0.12], [0.02, 0.04, 0.06]]], np.float32),
                            np.ones(3), np.ones(3), np.zeros((3, 3)), np.eye(3, dtype=np.float32),
                            0, {})
    settings = pipeline.Settings(balance="click", click_wb=(2.0, 1.0, 0.5), exposure=1.0,
                                 tone=(-8.0, 0.5, 2.5))
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.array([2.0, 1.0, 0.5]))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("linear rendering must not access color profiles or looks")

    monkeypatch.setattr(pipeline, "source_profile_for", unexpected)
    monkeypatch.setattr(pipeline, "resolve_space", unexpected)
    monkeypatch.setattr(pipeline, "color_lut", unexpected)
    monkeypatch.setattr(pipeline, "look_path", unexpected)
    monkeypatch.setattr(pipeline.profiles, "profile_dir", unexpected)
    monkeypatch.setattr(pipeline.icc, "trilinear", unexpected)

    image = pipeline.render(raw, settings, space="missing-space.icc", quality="faster", linear=True)

    before_tone = raw.camera_rgb * np.array([2.0, 1.0, 0.5], np.float32) * 2.0
    luminance = before_tone @ np.array([0.3, 0.59, 0.11], np.float32)
    ev = np.log2(np.maximum(luminance, 1e-7) / pipeline.MID_GREY)
    mapped_ev = np.interp(ev, settings.tone, pipeline.TONE_DEFAULT) + np.where(
        ev < settings.tone[0], ev - settings.tone[0], np.where(ev > settings.tone[2], ev - settings.tone[2], 0))
    expected = np.clip(before_tone * np.exp2(mapped_ev - ev).astype(np.float32)[..., None], 0, 1)
    np.testing.assert_allclose(image, expected, atol=1e-7)
    assert not np.allclose(image, raw.camera_rgb)


def test_render_linear_finishes_then_orients_and_crops_without_profiles(monkeypatch):
    from types import SimpleNamespace

    scene = np.repeat(np.arange(12, dtype=np.float32).reshape(3, 4, 1) / 20, 3, axis=2)
    raw = SimpleNamespace(caps=object(), scale=0.5, flip=0, lens_omen=None)
    settings = pipeline.Settings(rotation=1, crop=(1 / 3, 0.25, 1.0, 0.75))
    monkeypatch.setattr(pipeline, "scene_linear", lambda *_, **kwargs: kwargs == {"sensor_codes": True} and scene)
    seen = []

    def finish(image, received_settings, caps, scale, native_lens=False):
        seen.append((image, received_settings, caps, scale, native_lens))
        return image + 0.1

    monkeypatch.setattr(pipeline, "finish", finish)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("linear rendering must not access color profiles or looks")

    monkeypatch.setattr(pipeline, "source_profile_for", unexpected)
    monkeypatch.setattr(pipeline, "resolve_space", unexpected)
    monkeypatch.setattr(pipeline, "color_lut", unexpected)
    monkeypatch.setattr(pipeline, "look_path", unexpected)
    monkeypatch.setattr(pipeline.profiles, "profile_dir", unexpected)
    monkeypatch.setattr(pipeline.icc, "trilinear", unexpected)

    image = pipeline.render(raw, settings, quality="better", linear=True)

    expected = np.repeat(np.array([[5, 1], [6, 2]], np.float32)[..., None] / 20 + 0.1, 3, axis=2)
    np.testing.assert_allclose(image, expected)
    assert len(seen) == 1 and seen[0][0] is scene and seen[0][1] is settings
    assert seen[0][2] is raw.caps and seen[0][3:] == (raw.scale, False)


def _test_tiles(noise=0.0, blur=0.0, seed=0):
    """Analysis tiles: flat grey fields with chroma noise, beside step edges blurred by blur px."""
    import cv2
    rng = np.random.default_rng(seed)
    tiles = []
    for i in range(9):
        t = np.full((256, 256, 3), 0.45, np.float32)
        if i % 3 == 0:                                  # detailed tiles: a checkerboard of edges
            yy, xx = np.mgrid[0:256, 0:256]
            t[((yy // 32 + xx // 32) % 2) == 1] = 0.75
            if blur:
                t = cv2.GaussianBlur(t, (0, 0), blur)
        t = t + rng.normal(0, noise, (256, 256, 3)).astype(np.float32)
        tiles.append(np.clip(t, 0, 1))
    return tiles


def test_noise_auto_follows_measured_noise():
    caps = camera.capabilities("DCS Pro SLR/n")
    s = pipeline.Settings()
    quiet, noisy = _test_tiles(0.4 / 255), _test_tiles(3 / 255)
    assert pipeline.image_noise(noisy)[1] > 4 * pipeline.image_noise(quiet)[1]
    low = pipeline.noise_auto(quiet, s, caps, "basic")
    high = pipeline.noise_auto(noisy, s, caps, "basic")
    assert low["noise_type"] == high["noise_type"] == "basic"
    assert low["noise_strength"] < high["noise_strength"]
    assert pipeline.NOISE_RADIUS_RANGE["basic"][0] <= high["noise_radius"] <= pipeline.NOISE_RADIUS_RANGE["basic"][1]
    # Expert sets one strength, which also follows the noise; an older DCS image's Advanced with
    # Moire Reduction its threshold.
    ex_low, ex_high = pipeline.noise_auto(quiet, s, caps, "expert"), pipeline.noise_auto(noisy, s, caps, "expert")
    assert set(ex_high) == {"noise_type", "noise_strength", "noise_luma"}
    assert ex_low["noise_strength"] < ex_high["noise_strength"] and ex_high["noise_strength"] > 0
    dcs = camera.capabilities("DCS760C")
    assert set(pipeline.noise_auto(noisy, s, dcs, "moire")) == {"noise_type", "moire_threshold"}
    assert pipeline.noise_auto(noisy, s, caps, "none") == {"noise_type": "none"}
    # nothing flat or well exposed to measure
    dark = [np.zeros((256, 256, 3), np.float32)] * 9
    assert pipeline.noise_auto(dark, s, caps, "basic") is None


def test_expert_noise_reduction_follows_strength():
    caps = camera.capabilities("DCS Pro SLR/n")
    tiles = _test_tiles(3 / 255)
    flat = pipeline._flat_blocks(tiles)
    s = pipeline.Settings(noise_type="expert")

    def after(level):
        out = [pipeline.reduce_noise(t, s.copy(noise_strength=level, noise_luma=level), 1.0, caps) for t in tiles]
        return out, pipeline.image_noise(out, flat)

    noise = [after(level)[1] for level in (0, 25, 50, 75, 100)]
    assert noise[0] == pytest.approx(pipeline.image_noise(tiles, flat), rel=1e-3)   # 0 leaves it as is
    for a, b in zip(noise, noise[1:]):                                             # stronger, less noise
        assert b[0] < a[0] and b[1] < a[1]
    assert np.abs(after(0)[0][0] - tiles[0]).max() < 2e-3


def test_sharpen_auto_follows_edge_blur():
    s = pipeline.Settings()
    assert pipeline.edge_blur(_test_tiles(blur=1.5)) > pipeline.edge_blur(_test_tiles(blur=0.5))
    soft = pipeline.sharpen_auto(_test_tiles(blur=1.5), s)
    crisp = pipeline.sharpen_auto(_test_tiles(blur=0.3), s)
    assert soft[0] > crisp[0] and soft[1] > crisp[1]
    assert 0 < soft[0] <= pipeline.SHARPEN_PRESETS["high"][0]
    # the threshold rises with the noise so grain is not sharpened
    assert pipeline.sharpen_auto(_test_tiles(3 / 255, blur=1.5), s)[2] > soft[2]
    flat = [np.full((256, 256, 3), 0.5, np.float32)] * 9
    assert pipeline.sharpen_auto(flat, s) is None


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


def test_pro_slr_tone_lookup_has_bounded_monotonic_curve():
    lut = pipeline.pro_slr_tone_lut((-9.5, -0.2, 4.0))
    assert len(lut) == 4096
    assert np.all(np.isfinite(lut)) and np.all(np.diff(lut) >= 0)
    assert 0 <= lut[0] <= lut[-1] <= 1
    default = pipeline.pro_slr_tone_lut((-9.535300254821777, 0.0, 3.464400053024292))
    np.testing.assert_allclose(default[1:4090], np.arange(1, 4090) / 4095, atol=1 / 4095)
    assert lut[371] != pytest.approx(default[371], abs=1 / 4095)
    assert default[4091] == default[4090] == default[4095]


def test_tone_integer_table_rounded_boundaries_and_tail():
    identity = ((0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1))
    np.testing.assert_array_equal(pipeline.tone_integer_table(identity, end=9, output_max=7),
                                  [1, 1, 2, 3, 4, 5, 6, 7, 7, 7])
    np.testing.assert_array_equal(pipeline.tone_integer_table(identity, end=9, output_max=7, scale=2),
                                  [2, 2, 4, 6, 8, 10, 12, 14, 14, 14])
    np.testing.assert_array_equal(pipeline.tone_integer_table(((0, 0, 0), (2, 4, 0)), end=3, output_max=100),
                                  [1, 1, 4, 12])


def test_pro_slr_scene_applies_lookup_per_channel(monkeypatch):
    from types import SimpleNamespace
    raw = SimpleNamespace(camera_rgb=np.array([[[0.1, 0.3, 0.7]]], dtype=np.float32),
                          caps=SimpleNamespace(pro_slr=True))
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.array([1.0, 1.0, 1.0]))
    settings = pipeline.Settings(exposure=-1, tone=(-9.5, -0.2, 4.0))
    indices = np.floor(raw.camera_rgb * 0.5 * 4095 + 0.5).astype(np.uint16)
    np.testing.assert_array_equal(pipeline.scene_linear(raw, settings),
                                  pipeline.pro_slr_tone_lut(settings.tone)[indices])
    near_reference = list(pipeline.PRO_SLR_TONE_REFERENCE)
    near_reference[1] += 0.05
    unchanged = pipeline.scene_linear(raw, settings.copy(exposure=0, tone=tuple(near_reference)))
    np.testing.assert_array_equal(unchanged, raw.camera_rgb)
    disabled = settings.copy(exposure=0, tone_enabled=False)
    np.testing.assert_array_equal(pipeline.scene_linear(raw, disabled), raw.camera_rgb)
    assert disabled.copy(tone=(-9.5, -0.3, 4.0)).tone_enabled


def test_non_slr_scene_respects_tone_enable(monkeypatch):
    from types import SimpleNamespace
    raw = SimpleNamespace(camera_rgb=np.array([[[0.1, 0.3, 0.7]]], dtype=np.float32),
                          caps=SimpleNamespace(pro_slr=False))
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.ones(3))
    settings = pipeline.Settings(tone=(-9.5, -1.0, 3.4), tone_enabled=False)
    np.testing.assert_array_equal(pipeline.scene_linear(raw, settings), raw.camera_rgb)
    assert not np.allclose(pipeline.scene_linear(raw, settings.copy(tone_enabled=True)), raw.camera_rgb)


def test_auto_balance_region_uses_whole_scene(monkeypatch):
    import dataclasses
    @dataclasses.dataclass
    class Raw:
        camera_rgb: np.ndarray
        flip: int = 0
        balance_points: dict = dataclasses.field(default_factory=dict)
        caps: object = dataclasses.field(default_factory=lambda: type("Caps", (), {"pro_back": False})())
        scale: float = 1.0
        origin: tuple = (0, 0)
    scene = np.empty((64, 64, 3), dtype=np.float32)
    scene[:, :32] = (0.1, 0.5, 0.2)
    scene[:, 32:] = (0.7, 0.2, 0.3)
    raw = Raw(scene)
    seen = []
    def fake_render(sub, settings, *args):
        seen.append(settings.auto_balance_wb)
        return sub.camera_rgb
    monkeypatch.setattr(pipeline, "render", fake_render)
    settings = pipeline.Settings(balance="auto")
    pipeline.render_region(raw, settings, (0, 0, 0.5, 1), margin=0)
    np.testing.assert_allclose(seen[0], pipeline.auto_wb(scene))
    assert not np.allclose(seen[0], pipeline.auto_wb(scene[:, :32]))
    assert pipeline.Settings.from_json(settings.copy(auto_balance_wb=seen[0]).to_json()).auto_balance_wb == seen[0]


def test_auto_balance_uses_recorded_camera_point():
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    settings = pipeline.Settings(balance="auto")
    np.testing.assert_allclose(pipeline.method_wb(raw, settings), [2533 / 2048, 1, 2533 / 1798])
    assert pipeline.balance_temperature(raw, settings) == 4800
    assert pipeline.auto_exposure(raw, settings) == -1.0
    assert pipeline.auto_exposure(raw, settings.copy(color_balance=(100, 0, 0))) == -1.0
    raw.balance_curves.clear()
    assert pipeline.auto_kelvin_calibration(raw) is None
    assert pipeline.auto_exposure(raw, settings) == -1.0
    scene_settings = settings.copy(auto_balance_wb=(1.1, 1.0, 1.2))
    np.testing.assert_allclose(pipeline.method_wb(raw, scene_settings), [1.1, 1, 1.2])
    raw.balance_points["camera_auto"] = (4095, 4095, 4095)
    assert pipeline.camera_auto_wb(raw) is None


@pytest.mark.parametrize("balance,mode,low,high", [("auto", 5, 4000, 7500), ("custom", 4, 2500, 7000)])
@pytest.mark.parametrize("initial", [2000, 5200, 8000])
def test_pro_balance_mode_clamps_existing_temperature(balance, mode, low, high, initial):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow

    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    entry = SimpleNamespace(raw=raw, small=None)
    settings = pipeline.Settings(balance=balance, lighting="Daylight", white_point=initial,
                                 white_point_ref=initial, saved_balance_point=(2048, 2621, 1934))
    MainWindow._set_balance_temperature(None, entry, settings)
    assert settings.kelvin_mode == mode
    assert settings.kelvin_illumination == "Daylight"
    assert settings.white_point == settings.white_point_ref == max(low, min(high, initial))
    assert settings.saved_balance_point is None
    assert np.isfinite(pipeline.white_balance(raw, settings)).all()


def test_auto_scene_does_not_change_settings_when_decode_fails(monkeypatch):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    settings = pipeline.Settings(exposure=0.5)
    entry = SimpleNamespace(raw=None, small=None, path="missing.dcr",
                            caps=SimpleNamespace(auto_scene=True), settings=settings)
    def apply(entries, kind, change):
        candidate = entry.settings.copy()
        change(entry, candidate)
        assert candidate == settings
        return []
    sheet = SimpleNamespace(sheet=SimpleNamespace(apply=apply))
    window = SimpleNamespace(active_sheet=lambda: sheet, _raw_selection=lambda: [entry],
                             after_edit=lambda changed: None)
    monkeypatch.setattr(pipeline, "decode", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unreadable")))
    MainWindow.auto_scene(window)
    assert entry.settings is settings


def test_auto_exposure_changes_only_exposure():
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    original = pipeline.Settings(balance="custom", lighting="Tungsten", exposure=0.5, color_balance=(2, -3, 4))
    entry = SimpleNamespace(raw=raw, small=None, settings=original)
    def apply(entries, kind, change):
        candidate = entry.settings.copy()
        change(entry, candidate)
        assert candidate == original.copy(exposure=-1.0)
        return [entry]
    sheet = SimpleNamespace(sheet=SimpleNamespace(apply=apply))
    window = SimpleNamespace(active_sheet=lambda: sheet, _raw_selection=lambda: [entry],
                             after_edit=lambda changed: None)
    MainWindow.auto_exposure(window)


def test_pro_auto_exposure_does_not_require_kelvin_calibration():
    from types import SimpleNamespace
    raw = SimpleNamespace(caps=SimpleNamespace(pro_slr=True), auto_scene_exposure=-0.7)
    assert pipeline.auto_exposure(raw, pipeline.Settings()) == -0.7
    raw.auto_scene_exposure = None
    assert pipeline.auto_exposure(raw, pipeline.Settings()) is None


def test_auto_scene_retains_settings_when_calibration_unavailable():
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    raw.balance_curves.clear()
    original = pipeline.Settings(balance="preset", exposure=0.5)
    entry = SimpleNamespace(raw=raw, small=None, settings=original, caps=raw.caps)
    def apply(entries, kind, change):
        candidate = entry.settings.copy()
        change(entry, candidate)
        assert candidate == original
        return [entry]
    sheet = SimpleNamespace(sheet=SimpleNamespace(apply=apply))
    window = SimpleNamespace(active_sheet=lambda: sheet, _raw_selection=lambda: [entry],
                             after_edit=lambda changed: None)
    MainWindow.auto_scene(window)


def test_auto_scene_uses_auto_calibration_not_edited_temperature():
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    if not SLR_C_SAMPLE.is_file():
        pytest.skip("camera sample not available")
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    entry = SimpleNamespace(raw=raw, small=None, caps=raw.caps,
                            settings=pipeline.Settings(white_point=6500, white_point_ref=6500))
    result = []

    def apply(entries, kind, change):
        candidate = entry.settings.copy()
        change(entry, candidate)
        result.append(candidate)
        return [entry]

    sheet = SimpleNamespace(sheet=SimpleNamespace(apply=apply))
    window = SimpleNamespace(active_sheet=lambda: sheet, _raw_selection=lambda: [entry],
                             after_edit=lambda changed: None,
                             _set_balance_temperature=lambda e, s: MainWindow._set_balance_temperature(None, e, s))
    MainWindow.auto_scene(window)
    assert result[0].balance == "auto" and result[0].kelvin_mode == 5
    assert result[0].white_point == result[0].white_point_ref == 4800
    assert result[0].exposure == -1.0


def test_histogram_exposure_before_clipping(monkeypatch):
    from types import SimpleNamespace
    raw = SimpleNamespace(camera_rgb=np.full((2, 2, 3), 0.75, dtype=np.float32))
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.full(3, 2.0))
    settings = pipeline.Settings(exposure=-1.0)
    hist, edges = pipeline.ev_histogram(raw, settings, bins=160, lo=-4, hi=4)
    expected = np.log2(0.75 / pipeline.MID_GREY)
    assert hist.sum() == 1
    assert edges[np.argmax(hist)] <= expected < edges[np.argmax(hist) + 1]


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


def test_sheet_tracks_sidecar_origin_when_loading_refreshing_and_renaming(tmp_path, monkeypatch):
    from photodesk.gui.model import Sheet

    first = tmp_path / "first.DCR"
    first.write_bytes(b"")
    pipeline.save_settings(first, pipeline.Settings(white_point=6100, white_point_ref=6100))
    monkeypatch.setattr(camera, "read_make_model", lambda *_: ("Kodak", "test"))
    monkeypatch.setattr(pipeline, "load_settings", lambda *_: pipeline.Settings())

    sheet = Sheet(tmp_path)
    first_entry = sheet.find(first)
    assert first_entry.settings_from_sidecar

    second = tmp_path / "second.DCR"
    second.write_bytes(b"")
    sheet.refresh()
    second_entry = sheet.find(second)
    assert not second_entry.settings_from_sidecar

    third = tmp_path / "third.DCR"
    third.write_bytes(b"")
    pipeline.save_settings(third, pipeline.Settings())
    sheet.refresh()
    third_entry = sheet.find(third)
    assert third_entry.settings_from_sidecar

    sheet.rename([first_entry], ["renamed.DCR"])
    assert first_entry.settings_from_sidecar
    assert pipeline.sidecar_path(first_entry.path).is_file()


def test_thumbnail_temperature_initialization_respects_sidecar_and_async_edits(monkeypatch):
    from types import SimpleNamespace
    from photodesk.gui import app as gui_app
    from photodesk.gui.model import Entry

    tasks, temperatures, refreshed = [], [], []

    class FakeTask:
        def __init__(self, token, *_args):
            self.token = token
            self.signals = SimpleNamespace(
                done=SimpleNamespace(connect=lambda callback: setattr(self, "on_done", callback)),
                failed=SimpleNamespace(connect=lambda _callback: None),
            )
            tasks.append(self)

    monkeypatch.setattr(gui_app, "Task", FakeTask)
    monkeypatch.setattr(gui_app, "to_qimage", lambda image: image)
    monkeypatch.setattr(pipeline, "balance_temperature",
                        lambda _raw, settings: temperatures.append(settings) or 6200)

    window = SimpleNamespace(pdir="profiles", thumb_pool=SimpleNamespace(start=lambda _task: None),
                             focused=lambda: None, refresh_thumb=lambda entry: refreshed.append(entry))
    sw = SimpleNamespace(view=SimpleNamespace(refresh_entry=lambda _entry: None))

    def complete(settings, from_sidecar=False, edit_during_decode=None):
        entry = Entry(Path("image.DCR"), settings, settings_from_sidecar=from_sidecar)
        gui_app.MainWindow._load_thumb(window, sw, entry)
        if edit_during_decode is not None:
            entry.settings = edit_during_decode(entry.settings)
        tasks[-1].on_done(entry, (object(), {}, "thumbnail"))
        return entry

    computed = complete(pipeline.Settings(white_point=4800, white_point_ref=4800), from_sidecar=False)
    assert computed.settings.white_point == computed.settings.white_point_ref == 6200
    assert temperatures == [computed.settings]
    assert refreshed == [computed]

    sidecar = complete(pipeline.Settings(white_point=6300, white_point_ref=6300), from_sidecar=True)
    assert sidecar.settings.white_point == sidecar.settings.white_point_ref == 6300
    assert len(temperatures) == 1
    assert refreshed == [computed]

    restored = complete(pipeline.Settings(white_point=5000, white_point_ref=5000,
                                          kelvin_mode=0, kelvin_illumination="Daylight"))
    assert restored.settings.white_point == restored.settings.white_point_ref == 5000
    assert len(temperatures) == 1
    assert refreshed == [computed]

    edited = complete(pipeline.Settings(white_point=4700, white_point_ref=4700),
                      edit_during_decode=lambda settings: settings.copy(exposure=0.75))
    assert edited.settings.exposure == 0.75
    assert edited.settings.white_point == edited.settings.white_point_ref == 4700
    assert len(temperatures) == 1
    assert refreshed == [computed, edited]


def test_white_point_dialog_uses_100_kelvin_slider_steps():
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel, QSlider
    from photodesk.gui.dialogs import WhitePointDialog

    app = QApplication.instance() or QApplication([])
    window = SimpleNamespace(b={1212: QSlider(Qt.Horizontal), 1213: QLabel(), 1217: QLabel()},
                             changed=lambda: None)
    window._show = lambda notify=True: WhitePointDialog._show(window, notify)
    WhitePointDialog.setup(window, SimpleNamespace(white_point=5059))
    assert window.b[1213].text() == "5000° K"
    assert WhitePointDialog.values(window) == {"white_point": 5000}
    window.b[1212].setValue(51)
    assert window.b[1213].text() == "5100° K"
    assert WhitePointDialog.values(window) == {"white_point": 5100}


@pytest.mark.parametrize("mode,balance", [(4, "custom"), (5, "auto"), (6, "click")])
def test_lighting_edit_preserves_kelvin_mode_and_temperature(tmp_path, mode, balance):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    from photodesk.gui.model import Entry, Sheet

    settings = pipeline.Settings(balance=balance, lighting="Daylight", white_point=5100,
                                 white_point_ref=5100, kelvin_mode=mode,
                                 kelvin_illumination="Daylight", saved_balance_point=(2048, 2500, 1800))
    entry = Entry(tmp_path / "image.DCR", settings)
    sheet = Sheet(tmp_path)
    recalculated, changed = [], []
    window = SimpleNamespace(active_sheet=lambda: SimpleNamespace(sheet=sheet),
                             selection=lambda: [entry],
                             _set_balance_temperature=lambda e, s: recalculated.append((e, s)),
                             after_edit=lambda entries: changed.extend(entries))

    MainWindow.edit(window, "lighting", lighting="Daylight")
    assert entry.settings.to_json() == settings.to_json() and not changed and not recalculated
    MainWindow.edit(window, "lighting", lighting="Tungsten")
    assert changed == [entry] and not recalculated
    assert entry.settings.lighting == entry.settings.kelvin_illumination == "Tungsten"
    assert entry.settings.kelvin_mode == mode
    assert entry.settings.white_point == entry.settings.white_point_ref == 5100
    assert entry.settings.saved_balance_point is None
    sheet.undo()
    assert entry.settings.to_json() == settings.to_json()
    sheet.redo()
    assert entry.settings.lighting == "Tungsten" and entry.settings.kelvin_mode == mode
    copied = settings.copy(lighting="Tungsten")
    assert copied.kelvin_mode == mode and copied.kelvin_illumination == "Tungsten"
    assert copied.saved_balance_point is None and copied.white_point == 5100
    assert settings.copy(lighting="Tungsten", balance="preset").kelvin_mode is None


def test_balance_edit_keeps_original_settings_on_decode_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    from photodesk.gui.model import Entry, Sheet

    settings = pipeline.Settings(balance="preset", white_point=5100, white_point_ref=5100,
                                 saved_balance_point=(2048, 2500, 1800))
    entry = Entry(tmp_path / "unreadable.DCR", settings)
    sheet = Sheet(tmp_path)
    refreshed = []
    window = SimpleNamespace(active_sheet=lambda: SimpleNamespace(sheet=sheet),
                             selection=lambda: [entry],
                             _set_balance_temperature=lambda *_: pytest.fail("unexpected calibration"),
                             after_edit=lambda entries: refreshed.extend(entries))
    monkeypatch.setattr(pipeline, "decode", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))

    MainWindow.edit(window, "balance", balance="auto")

    assert entry.settings.to_json() == settings.to_json()
    assert not entry.dirty and not sheet.undo_stack and not refreshed


@pytest.mark.parametrize("command", ["balance_at_capture", "original_settings"])
def test_capture_reset_keeps_original_settings_on_decode_failure(tmp_path, monkeypatch, command):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    from photodesk.gui.model import Entry, Sheet

    settings = pipeline.Settings(balance="auto", exposure=0.5, white_point=5100)
    entry = Entry(tmp_path / "unreadable.DCR", settings)
    sheet = Sheet(tmp_path)
    refreshed = []
    window = SimpleNamespace(active_sheet=lambda: SimpleNamespace(sheet=sheet),
                             selection=lambda: [entry], _raw_selection=lambda: [entry],
                             selection_changed=lambda: None,
                             _set_balance_temperature=lambda *_: pytest.fail("unexpected calibration"),
                             after_edit=lambda entries: refreshed.extend(entries))
    monkeypatch.setattr(pipeline, "decode", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))

    getattr(MainWindow, command)(window)

    assert entry.settings.to_json() == settings.to_json()
    assert not entry.dirty and not sheet.undo_stack and not refreshed


def _capture_balance_raw():
    from types import SimpleNamespace

    curves = {}
    for illumination in ("Daylight", "Tungsten"):
        curve = [0.0] * 27
        curve[24], curve[25], curve[26] = 2500.0, 5000.0, 7500.0
        curves[illumination] = curve
    return SimpleNamespace(caps=camera.capabilities("DCS Pro SLR/c"),
                           auto_kelvin_illumination="Daylight", custom_kelvin_illumination="Tungsten",
                           balance_curves=curves)


@pytest.mark.skipif(not SLR_C_SAMPLE.is_file(), reason="Pro SLR/c sample not available")
def test_f5_capture_balance_uses_valid_recorded_auto_calibration():
    raw = pipeline.decode(SLR_C_SAMPLE, half_size=True)
    capture = pipeline.capture_settings(SLR_C_SAMPLE, raw)
    assert (capture.balance, capture.kelvin_mode, capture.kelvin_illumination) == ("auto", 5, "Daylight")
    assert capture.white_point == capture.white_point_ref == 4800
    assert capture.saved_balance_point is None
    assert np.isfinite(pipeline.white_balance(raw, capture)).all()


def _stub_capture_balance_metadata(monkeypatch, white_balance, source):
    monkeypatch.setattr(camera, "read_make_model", lambda _path: ("Kodak", "DCS Pro SLR/c"))
    monkeypatch.setattr(camera, "capture_info", lambda _path, model: {
        "model": model, "white_balance": white_balance,
    })
    monkeypatch.setattr(camera, "read_kodak_tags", lambda _path, tags: {1018: source} if 1018 in tags else {})


@pytest.mark.parametrize("command", ["balance_at_capture", "original_settings"])
@pytest.mark.parametrize("initial_temperature", [2300, 7200])
@pytest.mark.parametrize("white_balance,source,mode,illumination,calibrated,expected_temperature", [
    ("camera_auto", (1,), 5, "Daylight", "auto", 4800),
    ("custom", (0,), 4, "Tungsten", "custom", 6200),
])
def test_capture_reset_uses_independent_calibrated_balance(
        tmp_path, monkeypatch, command, initial_temperature, white_balance, source, mode, illumination,
        calibrated, expected_temperature):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow
    from photodesk.gui.model import Entry, Sheet

    raw = _capture_balance_raw()
    _stub_capture_balance_metadata(monkeypatch, white_balance, source)
    calls = []
    monkeypatch.setattr(pipeline, "auto_kelvin_calibration",
                        lambda _raw: calls.append("auto") or (4818, (1.0, 1.0, 1.0)))
    monkeypatch.setattr(pipeline, "custom_kelvin_calibration",
                        lambda _raw: calls.append("custom") or (6174, (1.0, 1.0, 1.0)))

    original = pipeline.Settings(
        balance="auto", lighting="Flash", white_point=initial_temperature, white_point_ref=initial_temperature,
        kelvin_mode=5, kelvin_illumination="Flash", saved_balance_point=(2048, 2500, 1800),
        auto_balance_wb=(1.2, 1.0, 0.9), exposure=0.75, color_balance=(2, -3, 4),
        sharpen_amount=17, tone=(-5.0, 0.0, 3.0), crop=(0.1, 0.2, 0.8, 0.9), rotation=2,
    )
    entry = Entry(tmp_path / "capture.DCR", original, model="DCS Pro SLR/c", raw=raw)
    sheet = Sheet(tmp_path)
    changed = []
    window = SimpleNamespace(active_sheet=lambda: SimpleNamespace(sheet=sheet), selection=lambda: [entry],
                             _raw_selection=lambda: [entry], after_edit=lambda entries: changed.extend(entries),
                             _set_balance_temperature=lambda *_: pytest.fail("calibrated capture used fallback"))
    window._restore_capture_balance = lambda e, s, cap: MainWindow._restore_capture_balance(window, e, s, cap)

    getattr(MainWindow, command)(window)

    result = entry.settings
    assert result.balance == calibrated
    assert result.lighting == result.kelvin_illumination == illumination
    assert result.kelvin_mode == mode
    assert result.white_point == result.white_point_ref == expected_temperature
    assert result.saved_balance_point is None and result.auto_balance_wb is None
    assert calls == [calibrated]
    assert changed == [entry]
    if command == "balance_at_capture":
        assert result.exposure == original.exposure
        assert result.color_balance == original.color_balance
        assert result.sharpen_amount == original.sharpen_amount
        assert result.tone == original.tone and result.crop == original.crop
        assert result.tone_enabled == original.tone_enabled
    else:
        assert result.exposure == 0.0 and result.tone == pipeline.PRO_SLR_TONE_REFERENCE
        assert not result.tone_enabled and result.crop == original.crop
    after = result.to_json()
    assert sheet.undo() == [entry]
    assert entry.settings.to_json() == original.to_json()
    assert sheet.redo() == [entry]
    assert entry.settings.to_json() == after


@pytest.mark.parametrize("source,calibration", [((2,), (4818, (1, 1, 1))), ((1, 1), (4818, (1, 1, 1))),
                                                  ((1,), None)])
def test_capture_auto_keeps_legacy_fallback_without_valid_calibration(tmp_path, monkeypatch, source, calibration):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow

    raw = _capture_balance_raw()
    _stub_capture_balance_metadata(monkeypatch, "camera_auto", source)
    monkeypatch.setattr(pipeline, "auto_kelvin_calibration", lambda _raw: calibration)
    capture = pipeline.capture_settings(tmp_path / "capture.DCR", raw)
    assert capture.balance == "preset" and capture.kelvin_mode is None
    # The no-RAW API retains its historical tag-1020 mapping rather than guessing Auto.
    assert pipeline.capture_settings(tmp_path / "capture.DCR").balance == "preset"

    settings = pipeline.Settings(balance="custom", white_point=6800, white_point_ref=6800)
    fallback = []
    window = SimpleNamespace(_set_balance_temperature=lambda _e, s: fallback.append(s.white_point))
    MainWindow._restore_capture_balance(window, object(), settings, capture)
    assert settings.balance == "preset" and fallback == [6800]


def _grbg_cfa(shape, r, g, b, lens_omen=None, origin=(0, 0)):
    codes = np.empty(shape, np.uint16)
    codes[0::2, 0::2] = g
    codes[1::2, 1::2] = g
    codes[0::2, 1::2] = r
    codes[1::2, 0::2] = b
    return pipeline.SensorCFA(codes, np.array([[1, 0], [2, 1]]), 0, lens_omen, origin)


def test_pro_slr_click_rectangle_is_even_aligned_and_bounded():
    # Unique codes per site reveal which 16x16 rectangle the trimmed means came from.
    cfa = _grbg_cfa((32, 34), 0, 0, 0)
    ys, xs = np.mgrid[0:32, 0:34]
    cfa.codes[:] = 100 + ys * 34 + xs

    def rect(x0, y0):
        patch = cfa.codes[y0:y0 + 16, x0:x0 + 16].astype(np.int64)
        means = []
        for sites in (patch[0::2, 1::2].ravel(), np.concatenate([patch[0::2, 0::2].ravel(), patch[1::2, 1::2].ravel()]),
                      patch[1::2, 0::2].ravel()):
            v = np.sort(sites)
            means.append(math.floor(v[len(v) // 4:len(v) - len(v) // 4].mean() + 0.5))
        return tuple(means)

    assert pipeline.click_point_pro_slr(cfa, 17, 17) == rect(10, 10)
    assert pipeline.click_point_pro_slr(cfa, 16, 16) == rect(8, 8)
    assert pipeline.click_point_pro_slr(cfa, 0, 0) == rect(0, 0)
    assert pipeline.click_point_pro_slr(cfa, 33, 31) == rect(18, 16)


def test_pro_slr_click_balance_trims_cfa_sites_with_native_code_thresholds():
    cfa = _grbg_cfa((32, 32), 1000, 2000, 500)
    cfa.codes[10, 11] = 4000          # an R outlier is trimmed away
    cfa.codes[11, 10] = 0             # so is a dead B site
    assert pipeline.click_point_pro_slr(cfa, 15, 15) == (1000, 2000, 500)
    np.testing.assert_allclose(pipeline.click_wb_pro_slr(cfa, 15, 15), [2.0, 1.0, 4.0])

    assert pipeline.click_point_pro_slr(_grbg_cfa((16, 16), 0xc80, 0xc80, 0x32), 8, 8) == (0xc80, 0xc80, 0x32)
    with pytest.raises(pipeline.ClickBalanceError, match="saturated"):
        pipeline.click_wb_pro_slr(_grbg_cfa((16, 16), 0xc81, 2000, 500), 8, 8)
    with pytest.raises(pipeline.ClickBalanceError, match="too dark"):
        pipeline.click_wb_pro_slr(_grbg_cfa((16, 16), 1000, 2000, 0x31), 8, 8)


def test_pro_slr_click_balance_applies_requested_lens_optimization_to_blue_sites():
    grid = np.full((48, 71), 0x1200, np.int64)        # blue falloff: x = 0x200 everywhere
    omen = {"grid": grid, "unit": 0x1000, "scale": (0.0, 1.0), "baseline": 0}
    cfa = _grbg_cfa((32, 32), 1000, 2000, 1000, omen, origin=(6, 8))
    assert pipeline.click_point_pro_slr(cfa, 15, 15) == (1000, 2000, 1000)
    assert pipeline.click_point_pro_slr(cfa, 15, 15, lens_optimization=0) == (1000, 2000, 1000)
    gain = int(demosaic.omen_tiles(grid, 0x1000, (0.0, 1.0), 50, 0)[0, 0])
    assert gain > 4096
    expected_b = (1000 * gain + 0x800) >> 12
    assert pipeline.click_point_pro_slr(cfa, 15, 15, lens_optimization=50) == (1000, 2000, expected_b)

    # Other camera families keep the legacy ordinary patch mean, including its outlier influence.
    generic = np.full((5, 5, 3), (0.4, 0.2, 0.1), dtype=np.float32)
    generic[2, 2] = (0.7, 0.8, 0.9)
    mean = generic.mean((0, 1))
    np.testing.assert_allclose(pipeline.click_wb(generic, 2, 2, radius=2), mean[1] / mean)


def test_click_balance_does_not_edit_an_unselected_reference(monkeypatch):
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow

    raw = SimpleNamespace(camera_rgb=np.ones((16, 16, 3), np.float32), flip=0)
    reference = SimpleNamespace(is_raw=True, raw=raw, small=raw, serial="camera-a",
                                settings=SimpleNamespace(rotation=0))
    selected = SimpleNamespace(is_raw=True, serial="camera-b")
    edits = []
    clicks = []
    window = SimpleNamespace(selection=lambda: [selected], edit=lambda *args, **kw: edits.append((args, kw)))

    def generic_click(cam, x, y, radius):
        clicks.append((cam, x, y, radius))
        return 1.0, 1.0, 1.0

    monkeypatch.setattr(pipeline, "click_wb", generic_click)

    MainWindow.click_balance(window, reference, 0.5, 0.5)

    assert len(edits) == 1 and edits[0][0][1] == []
    assert clicks[0][1:] == (8, 8, 1)
    matching = SimpleNamespace(is_raw=True, serial="camera-a")
    window.selection = lambda: [selected, matching]
    MainWindow.click_balance(window, reference, 0.5, 0.5)
    assert edits[-1][0][1] == [matching]
    assert clicks[-1][1:] == (8, 8, 1)
    reference.serial = None
    unknown = SimpleNamespace(is_raw=True, serial=None)
    window.selection = lambda: [reference, unknown]
    MainWindow.click_balance(window, reference, 0.5, 0.5)
    assert edits[-1][0][1] == [reference]


def test_pro_slr_thumbnail_click_uses_full_decode_and_preserves_camera_selection(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from photodesk.gui import app as gui_app
    from photodesk.gui.app import MainWindow

    callbacks = {}

    class Signal:
        def connect(self, callback):
            self.callback = callback

        def emit(self, *args):
            self.callback(*args)

    class Task:
        def __init__(self, token, fn, *args):
            self.token, self.fn, self.args = token, fn, args
            self.signals = SimpleNamespace(done=Signal(), failed=Signal())

    class Pool:
        def start(self, task, priority):
            callbacks["priority"] = priority
            task.signals.done.emit(task.token, task.fn(*task.args))

    small = SimpleNamespace(camera_rgb=np.zeros((10, 12, 3), np.float32), flip=6,
                            caps=SimpleNamespace(pro_slr=True))
    entry = SimpleNamespace(is_raw=True, raw=None, small=small, serial="camera-a", path=tmp_path / "x.DCR",
                            settings=SimpleNamespace(rotation=1, lens_optimization=10))
    matching = SimpleNamespace(is_raw=True, serial="camera-a")
    other = SimpleNamespace(is_raw=True, serial="camera-b")
    sheet = object()
    edits = []
    window = SimpleNamespace(selection=lambda: [matching, other], active_sheet=lambda: sheet,
                             edit=lambda *args, **kwargs: edits.append((args, kwargs)), thumb_pool=Pool())

    def read_cfa(path):
        callbacks["read"] = path
        return SimpleNamespace(codes=np.zeros((40, 60), np.uint16), flip=6)

    def sample(cfa, x, y, lens):
        callbacks["sample"] = (cfa.codes.shape, x, y, lens)
        return (1.25, 1.0, 0.75)

    monkeypatch.setattr(gui_app, "Task", Task)
    monkeypatch.setattr(pipeline, "read_cfa", read_cfa)
    monkeypatch.setattr(pipeline, "click_wb_pro_slr", sample)

    MainWindow.click_balance(window, entry, 0.25, 0.75, thumb=True)

    assert callbacks["read"] == entry.path
    assert callbacks["sample"] == ((40, 60), 45, 10, 10)
    assert callbacks["priority"] == 1
    assert edits == [(("click_balance", [matching]), {"balance": "click", "click_wb": (1.25, 1.0, 0.75)})]


def test_pro_slr_click_discards_older_async_result(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from photodesk.gui import app as gui_app

    tasks, edits = [], []

    class Signal:
        def connect(self, callback):
            self.callback = callback

    class Task:
        def __init__(self, token, *_args):
            self.token = token
            self.signals = SimpleNamespace(done=Signal(), failed=Signal())
            tasks.append(self)

    raw = SimpleNamespace(camera_rgb=np.zeros((20, 20, 3), np.float32), flip=0,
                          caps=SimpleNamespace(pro_slr=True))
    entry = SimpleNamespace(is_raw=True, raw=raw, small=raw, serial="camera-a", path=tmp_path / "x.DCR",
                            settings=SimpleNamespace(rotation=0, lens_optimization=0))
    sheet = object()
    window = SimpleNamespace(selection=lambda: [entry], active_sheet=lambda: sheet,
                             edit=lambda *args, **kwargs: edits.append((args, kwargs)),
                             thumb_pool=SimpleNamespace(start=lambda *_: None))
    monkeypatch.setattr(gui_app, "Task", Task)

    gui_app.MainWindow.click_balance(window, entry, 0.25, 0.25)
    gui_app.MainWindow.click_balance(window, entry, 0.75, 0.75)
    tasks[1].signals.done.callback(tasks[1].token, (1.2, 1.0, 1.1))
    tasks[0].signals.done.callback(tasks[0].token, (1.4, 1.0, 0.9))
    assert len(edits) == 1 and edits[0][1]["click_wb"] == (1.2, 1.0, 1.1)
    gui_app.MainWindow.click_balance(window, entry, 0.5, 0.5)
    window.refresh_thumb = lambda _entry: None
    window.sheets = []
    window.closeups = []
    window.selection_changed = lambda: None
    gui_app.MainWindow.after_edit(window, [entry])
    tasks[2].signals.done.callback(tasks[2].token, (1.1, 1.0, 1.4))
    assert len(edits) == 1


def test_preview_refresh_updates_matching_closeups():
    from types import SimpleNamespace
    from photodesk.gui.app import MainWindow

    entry, other = object(), object()
    matching, unrelated = SimpleNamespace(entry=entry), SimpleNamespace(entry=other)
    calls = []
    window = SimpleNamespace(sheets=[], closeups=[matching, unrelated],
                             refresh_thumb=lambda item: calls.append(("thumb", item)),
                             _update_closeup=lambda item: calls.append(("closeup", item)))
    MainWindow._preview_refresh(window, [entry])
    assert calls == [("thumb", entry), ("closeup", matching)]


def test_closeup_discards_stale_preview_results(monkeypatch):
    from types import SimpleNamespace
    from photodesk.gui import app as gui_app

    tasks, displayed = [], []

    class FakeTask:
        def __init__(self, *args):
            self.signals = SimpleNamespace(done=SimpleNamespace(connect=lambda callback: setattr(self, "done", callback)))
            tasks.append(self)

    raw = SimpleNamespace(camera_rgb=np.zeros((8, 8, 3), np.float32), flip=0)
    entry = SimpleNamespace(is_raw=True, raw=raw, meta={}, settings=SimpleNamespace(rotation=0))
    closeup = SimpleNamespace(entry=entry, region=lambda size: (0, 0, 1, 1))
    window = SimpleNamespace(opts=SimpleNamespace(working_space="test"), pdir="test",
                             pool=SimpleNamespace(start=lambda task: None),
                             _closeup_ready=lambda cu, image: displayed.append(image))
    monkeypatch.setattr(gui_app, "Task", FakeTask)
    monkeypatch.setattr(pipeline, "resolve_space", lambda *args: "test")

    gui_app.MainWindow._update_closeup(window, closeup)
    gui_app.MainWindow._update_closeup(window, closeup)
    tasks[1].done(closeup, "new")
    tasks[0].done(closeup, "old")
    assert displayed == ["new"]


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


def test_ev_histogram_curve_data_handles_available_and_missing_data():
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from photodesk.gui.views import EVHistogram

    app = QApplication.instance() or QApplication([])
    widget = EVHistogram()
    ev = np.array([-12.0, -3.0, 6.0], np.float32)
    output = np.array([0.02, 0.4, 0.98], np.float32)
    widget.resize(258, 100)
    widget.set_curve((ev, output))
    np.testing.assert_array_equal(widget.curve[0], ev)
    np.testing.assert_array_equal(widget.curve[1], output)
    image = QImage(widget.size(), QImage.Format.Format_RGB32)
    widget.render(image)
    assert any(image.pixelColor(x, y).blue() > image.pixelColor(x, y).red() + 100
               for x in range(1, 257) for y in range(1, 85))
    widget.set_curve(None)
    assert widget.curve is None
    widget.set_curve(([0.0, 1.0], [0.5]))
    assert widget.curve is None


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


def test_working_space_survives_app_move(tmp_path):
    # macOS runs a downloaded app from a new random path each launch (App Translocation): a stored
    # bundled profile must be found again in the current profile directory.
    old, new = tmp_path / "old/profiles", tmp_path / "new/profiles"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "KODAK_sRGB.icm").write_bytes(b"x")
        (d / pipeline.DEFAULT_WORKING_SPACE).write_bytes(b"x")
    assert pipeline.space_setting(str(new / "KODAK_sRGB.icm"), new) == "KODAK_sRGB.icm"
    assert pipeline.resolve_space("KODAK_sRGB.icm", new) == str(new / "KODAK_sRGB.icm")
    stale = str(tmp_path / "gone/profiles/KODAK_sRGB.icm")
    assert pipeline.resolve_space(stale, new) == str(new / "KODAK_sRGB.icm")
    # a profile outside the profile directory keeps its path; an unknown one falls back to sRGB
    assert pipeline.space_setting(str(old / "KODAK_sRGB.icm"), new) == str(old / "KODAK_sRGB.icm")
    assert pipeline.resolve_space(str(tmp_path / "gone/Missing.icc"), new) == str(new / pipeline.DEFAULT_WORKING_SPACE)


@need_sample
def test_decode_sample(raw):
    assert raw.make == "Kodak" and raw.model == "ProBack645"
    assert raw.camera_rgb.shape == (2040, 2040, 3)
    assert 0 < raw.camera_rgb.max() <= 1


def _fake_decode(monkeypatch, model, sizes, image):
    from types import SimpleNamespace

    class FakeRaw:
        def __init__(self):
            self.sizes = sizes
            self.image = image
            self.other = SimpleNamespace(iso_speed=100, shutter_speed=1, aperture=2.8,
                                         focal_length=50, timestamp=None, artist="")
            self.daylight_whitebalance = [1, 1, 1, 1]
            self.camera_whitebalance = [1, 1, 1, 1]
            self.rgb_xyz_matrix = np.eye(3)
            self.white_level = 4095
            self.params = None
            # An R G / G B layout, which the native Kodak demosaic declines.
            self.raw_pattern = np.array([[0, 1], [3, 2]])
            self.color_desc = b"RGBG"
            self.black_level_per_channel = [0, 0, 0, 0]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def postprocess(self, **params):
            self.params = params
            return self.image.copy()

    raw = FakeRaw()
    monkeypatch.setattr(pipeline.camera, "read_make_model", lambda _path: ("Kodak", model))
    monkeypatch.setattr(pipeline.rawpy, "imread", lambda _path: raw)
    monkeypatch.setattr(pipeline.camera, "read_gps", lambda _path: None)
    monkeypatch.setattr(pipeline.camera, "auto_capture_settings", lambda _path: (None, None))
    monkeypatch.setattr(pipeline.camera, "balance_points", lambda _path: {})
    monkeypatch.setattr(pipeline.camera, "balance_curves", lambda _path: {})
    monkeypatch.setattr(pipeline.camera, "balance_factors", lambda _path: {})
    monkeypatch.setattr(pipeline.camera, "color_matrices", lambda _path: {})
    monkeypatch.setattr(pipeline.camera, "unique_matrices", lambda _path: {})
    monkeypatch.setattr(pipeline.camera, "lens_omen", lambda *_: None)
    monkeypatch.setattr(pipeline.camera, "custom_kelvin_illumination", lambda _path: None)
    return raw


def _coordinate_image(height, width):
    y, x = np.indices((height, width))
    return np.stack((x, y, x + y), axis=-1).astype(np.uint16)


def _normalise_test_image(image, model="DCS Pro SLR/n"):
    image = image.astype(np.float32)
    image *= np.float32(1 / 65535)
    if model in pipeline.PRO_SLR_SENSOR_GAINS:
        image *= pipeline.PRO_SLR_SENSOR_GAINS[model]
    return image


@pytest.mark.parametrize("half_size,shape,expected_box,expected_scale", [
    (False, (10, 12), (2, 2, 10, 8), 1.0),
    (True, (5, 6), (1, 1, 5, 4), 0.5),
])
def test_decode_crops_pro_slr_full_and_half_size(monkeypatch, half_size, shape, expected_box,
                                                 expected_scale):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=12, raw_height=10, width=12, height=10,
                            left_margin=0, top_margin=0, crop_left_margin=2,
                            crop_top_margin=2, crop_width=8, crop_height=6, flip=0)
    image = _coordinate_image(*shape)
    fake = _fake_decode(monkeypatch, "DCS Pro SLR/n", sizes, image)
    decoded = pipeline.decode("synthetic.dcr", half_size=half_size)
    x0, y0, x1, y1 = expected_box

    assert fake.params["half_size"] is half_size
    assert decoded.camera_rgb.shape == (y1 - y0, x1 - x0, 3)
    np.testing.assert_array_equal(decoded.camera_rgb, _normalise_test_image(image[y0:y1, x0:x1]))
    assert decoded.metadata["width"] == 8 and decoded.metadata["height"] == 6
    assert decoded.scale == expected_scale


def test_decode_demosaics_full_size_pro_slr_n_with_balanced_kodak_ah2(monkeypatch):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=12, raw_height=10, width=12, height=10,
                            left_margin=0, top_margin=0, crop_left_margin=2,
                            crop_top_margin=2, crop_width=8, crop_height=6, flip=0)
    fake = _fake_decode(monkeypatch, "DCS Pro SLR/n", sizes, _coordinate_image(10, 12))
    fake.raw_pattern = np.array([[1, 0], [2, 3]])   # G R / B G at the crop origin
    fake.color_desc = b"RGBG"
    fake.white_level = 3700
    fake.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
    cfa = np.zeros((10, 12), np.uint16)
    cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = 900, 500, 1000, 900
    cfa[0, 1] = 3000   # outside the crop
    fake.raw_image_visible = cfa
    decoded = pipeline.decode("synthetic.dcr")

    assert fake.params is None   # LibRaw postprocess is not used
    assert decoded.camera_rgb.shape == (6, 8, 3)
    # Flat field: every pixel gets the site samples back, unbalanced and white-level normalised.
    expected = np.array([500, 900, 1000], np.float32) / 3700 * pipeline.PRO_SLR_SENSOR_GAINS["DCS Pro SLR/n"]
    np.testing.assert_allclose(decoded.camera_rgb, np.broadcast_to(expected, (6, 8, 3)), rtol=1e-6)
    # Half-size previews still bin through LibRaw.
    pipeline.decode("synthetic.dcr", half_size=True)
    assert fake.params["half_size"] is True


@pytest.mark.parametrize("shape", [(4, 5), (5, 6)])
def test_decode_maps_odd_half_size_crop_to_actual_output_bounds(monkeypatch, shape):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=11, raw_height=9, width=11, height=9,
                            left_margin=0, top_margin=0, crop_left_margin=3,
                            crop_top_margin=1, crop_width=6, crop_height=6, flip=0)
    image = _coordinate_image(*shape)
    _fake_decode(monkeypatch, "DCS Pro SLR/n", sizes, image)
    decoded = pipeline.decode("synthetic.dcr", half_size=True)

    np.testing.assert_array_equal(decoded.camera_rgb,
                                  _normalise_test_image(image[0:4, 1:5]))
    assert decoded.camera_rgb.shape == (4, 4, 3)
    assert decoded.metadata["width"] == 6 and decoded.metadata["height"] == 6
    assert decoded.scale == pytest.approx(2 / 3)


def test_decode_crop_offsets_are_relative_to_visible_area(monkeypatch):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=14, raw_height=10, width=8, height=6,
                            left_margin=3, top_margin=2, crop_left_margin=5,
                            crop_top_margin=4, crop_width=4, crop_height=2, flip=0)
    image = _coordinate_image(6, 8)
    _fake_decode(monkeypatch, "DCS Pro SLR/c", sizes, image)
    decoded = pipeline.decode("synthetic.dcr")

    np.testing.assert_array_equal(decoded.camera_rgb,
                                  _normalise_test_image(image[2:4, 2:6], "DCS Pro SLR/c"))
    assert decoded.metadata["width"] == 4 and decoded.metadata["height"] == 2


def test_decode_reduction_and_region_preserve_lens_record_and_origin(monkeypatch):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=10, raw_height=8, width=10, height=8, flip=0)
    _fake_decode(monkeypatch, "DCS Pro SLR/n", sizes, _coordinate_image(8, 10))
    record = _omen_record()
    monkeypatch.setattr(pipeline.camera, "lens_omen", lambda *_: record)
    raw = pipeline.decode("synthetic.dcr")
    assert raw.lens_omen is record
    assert pipeline.reduced(raw, 4).lens_omen is record

    seen = []

    def render(sub, settings, *args, **kwargs):
        seen.append((sub.lens_omen, sub.origin))
        return sub.camera_rgb

    monkeypatch.setattr(pipeline, "render", render)
    pipeline.render_region(raw, pipeline.Settings(sharpen_amount=0), (0.5, 0.5, 1, 1), margin=0)
    assert seen == [(record, (raw.origin[0] + 4, raw.origin[1] + 5))]


@pytest.mark.parametrize("crop_values", [None, (2, 2, 0, 6), (10, 2, 4, 6)])
def test_decode_ignores_missing_or_invalid_pro_slr_crop(monkeypatch, crop_values):
    from types import SimpleNamespace

    values = dict(raw_width=12, raw_height=10, width=12, height=10,
                  left_margin=0, top_margin=0, flip=0)
    if crop_values is not None:
        values.update(zip(("crop_left_margin", "crop_top_margin", "crop_width", "crop_height"),
                          crop_values))
    sizes = SimpleNamespace(**values)
    image = _coordinate_image(10, 12)
    _fake_decode(monkeypatch, "DCS Pro SLR/n", sizes, image)
    decoded = pipeline.decode("synthetic.dcr")

    np.testing.assert_array_equal(decoded.camera_rgb, _normalise_test_image(image))
    assert decoded.metadata["width"] == 12 and decoded.metadata["height"] == 10


@pytest.mark.parametrize("model", ["DCS Pro 14n", "DCS660M"])
def test_decode_leaves_other_model_families_uncropped(monkeypatch, model):
    from types import SimpleNamespace

    sizes = SimpleNamespace(raw_width=12, raw_height=10, width=12, height=10,
                            left_margin=0, top_margin=0, crop_left_margin=2,
                            crop_top_margin=2, crop_width=8, crop_height=6, flip=0)
    image = _coordinate_image(10, 12)
    _fake_decode(monkeypatch, model, sizes, image)
    decoded = pipeline.decode("synthetic.dcr")

    assert decoded.camera_rgb.shape == image.shape
    assert decoded.metadata["width"] == 12 and decoded.metadata["height"] == 10


def test_pro_slr_crop_matches_optional_tiff_reference():
    import tifffile

    reference = ROOT / "raw/sample_tiff/GQ432034.tif"
    source = SLR_N_SAMPLE
    if not reference.is_file() or not source.is_file():
        pytest.skip("Pro SLR/n sample RAW or golden TIFF not available")
    with tifffile.TiffFile(reference) as tiff:
        reference_height, reference_width = tiff.pages[0].shape[:2]
    assert (reference_width, reference_height) == (4500, 3000)

    full = pipeline.decode(source)
    half = pipeline.decode(source, half_size=True)
    assert (full.metadata["width"], full.metadata["height"]) == (reference_width, reference_height)
    assert full.camera_rgb.shape[:2] == (reference_height, reference_width)
    assert half.camera_rgb.shape[:2] == (reference_height // 2, reference_width // 2)
    assert half.scale == 0.5


def test_linear_render_matches_optional_native_tiff_samples():
    import cv2
    import tifffile

    # Original Photo Desk Linear Mode exports of GQ432034.DCR: Daylight preset at 5000 K, Product
    # Look, no lens optimization, noise reduction or sharpening, EV 0 and +1, with tone
    # (-9.5, -0.2, 4.0) and the neutral tone (-9.5, 0.0, 3.5).
    cases = (("linear_ev0.tif", 0.0, (-9.5, -0.2, 4.0)), ("linear_ev1.tif", 1.0, (-9.5, -0.2, 4.0)),
             ("linear_neutral_ev0.tif", 0.0, (-9.5, 0.0, 3.5)),
             ("linear_neutral_ev1.tif", 1.0, (-9.5, 0.0, 3.5)))
    references = [(ROOT / "raw/sample_tiff" / name, exposure, tone) for name, exposure, tone in cases]
    if not SLR_N_SAMPLE.is_file() or not all(path.is_file() for path, *_ in references):
        pytest.skip("Pro SLR/n sample RAW and the linear TIFF references are required")

    raw = pipeline.decode(SLR_N_SAMPLE, half_size=True)
    settings = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000,
                                 white_point_ref=5000, sharpen_amount=0)
    for reference, exposure, tone in references:
        rmse_limit = 0.006
        with tifffile.TiffFile(reference) as tiff:
            source = tiff.asarray()
        assert source.dtype == np.uint16 and source.shape == (
            raw.camera_rgb.shape[0] * 2, raw.camera_rgb.shape[1] * 2, 3)
        expected = cv2.resize(source.astype(np.float32) / 65535.0,
                              (raw.camera_rgb.shape[1], raw.camera_rgb.shape[0]),
                              interpolation=cv2.INTER_AREA)

        image = pipeline.render(raw, settings.copy(exposure=exposure, tone=tone), space="missing-space.icc",
                                profile_directory="missing-profile-directory", quality="faster", linear=True)
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        # Measured RMSE is about .003-.005 (half-size binning instead of the Kodak demosaic).
        assert rmse < rmse_limit, f"{reference.name} linear-domain RMSE {rmse:.5f} exceeds {rmse_limit}"


def test_linear_render_matches_optional_native_control_samples(tmp_path):
    import cv2
    import tifffile

    # Original Linear Mode exports of GQ432034.DCR, each changing one control from Daylight preset
    # 5000 K, EV 0, tone (-9.5, -0.2, 4.0), colour balance 0, no lens/noise/sharpening.
    cases = (("minus1.5.tif", {"exposure": -1.5}), ("plus2.0.tif", {"exposure": 2.0}),
             ("4000K.tif", {"white_point": 4000, "white_point_ref": 4000}),
             ("7500K.tif", {"white_point": 7500, "white_point_ref": 7500}),
             ("R50B-50.tif", {"color_balance": (50, 0, -50)}),
             ("tone_s-8_m+8_h+3.tif", {"tone": (-8.0, 0.5, 3.0)}),
             ("텅스텐.tif", {"lighting": "Tungsten", "white_point": 3200, "white_point_ref": 3200}),
             ("형광등.tif", {"lighting": "Fluorescent"}),
             ("플래쉬.tif", {"lighting": "Flash", "white_point": 5500, "white_point_ref": 5500}))
    folder = ROOT / "raw/sample_tiff/test1"
    if not SLR_N_SAMPLE.is_file() or not all((folder / name).is_file() for name, _ in cases):
        pytest.skip("Pro SLR/n sample RAW and the control TIFF references are required")
    raw = pipeline.decode(SLR_N_SAMPLE, half_size=True)
    settings = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000,
                                 white_point_ref=5000, tone=(-9.5, -0.2, 4.0), sharpen_amount=0)
    for name, fields in cases:
        expected = cv2.resize(tifffile.imread(folder / name).astype(np.float32) / 65535.0,
                              (raw.camera_rgb.shape[1], raw.camera_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        image = pipeline.render(raw, settings.copy(**fields), quality="faster", linear=True)
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        # Measured RMSE is .0013-.0045; half-size previews bin the CFA instead of demosaicing it.
        assert rmse < 0.006, f"{name} linear-domain RMSE {rmse:.5f}"


def test_linear_render_matches_optional_native_balance_mode_samples():
    import cv2
    import tifffile

    folder = ROOT / "raw/sample_tiff/test2"
    names = ("자동.tif", "커스텀.tif", "클릭.tif")
    if not SLR_N_SAMPLE.is_file() or not all((folder / name).is_file() for name in names):
        pytest.skip("Pro SLR/n sample RAW and the balance-mode TIFF references are required")
    raw = pipeline.decode(SLR_N_SAMPLE, half_size=True)
    base = pipeline.Settings(lighting="Daylight", white_point=5000, white_point_ref=5000,
                             tone=(-9.5, -0.2, 4.0), sharpen_amount=0)
    auto_temperature, custom_temperature = 5000, 5800   # the app's Auto/Custom proposals here
    cases = (
        # Native Auto keeps the current 5000 K over the camera Auto calibration; the remaining
        # gap (about .005 versus a best-fit .0026) is an unresolved calibration difference.
        (names[0], base.copy(balance="auto", kelvin_mode=5, kelvin_illumination="Daylight",
                             white_point=auto_temperature, white_point_ref=auto_temperature), 0.006),
        (names[1], base.copy(balance="custom", kelvin_mode=4, kelvin_illumination="Daylight",
                             white_point=custom_temperature, white_point_ref=custom_temperature), 0.004),
        # Click Balance at image pixel (837, 892), as shown in Photo Desk's status bar.
        (names[2], base.copy(balance="click", click_wb=pipeline.click_wb_pro_slr(pipeline.read_cfa(SLR_N_SAMPLE), 837, 892, 0)),
         0.003),
    )
    for name, settings, limit in cases:
        expected = cv2.resize(tifffile.imread(folder / name).astype(np.float32) / 65535.0,
                              (raw.camera_rgb.shape[1], raw.camera_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        image = pipeline.render(raw, settings, quality="faster", linear=True)
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        assert rmse < limit, f"{name} linear-domain RMSE {rmse:.5f}"


def test_pro_slr_scene_applies_camera_matrix_after_clipped_balance(monkeypatch):
    from types import SimpleNamespace
    matrix = (1.25, -0.25, 0.0, 0.0, 1.0, 0.0, -0.5, 0.0, 1.5)
    raw = SimpleNamespace(camera_rgb=np.array([[[0.4, 0.2, 0.8]]], np.float32),
                          caps=SimpleNamespace(pro_slr=True), white_level=3700.0,
                          color_matrices={"Daylight": matrix})
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.array([1.0, 1.0, 1.5]))
    settings = pipeline.Settings(lighting="Daylight", tone_enabled=False)
    balanced = np.clip(raw.camera_rgb * np.float32([1.0, 1.0, 1.5]), 0, 1)   # blue clips at 1
    expected = np.clip(balanced @ np.array(matrix, np.float32).reshape(3, 3).T, 0, 1)
    np.testing.assert_allclose(pipeline.scene_linear(raw, settings), expected, rtol=1e-6)
    # Linear Mode expresses the 3700-white samples on Kodak's 12-bit code scale.
    codes = np.clip(raw.camera_rgb * np.float32([1.0, 1.0, 1.5]) * np.float32(3700 / 4095), 0, 1)
    np.testing.assert_allclose(pipeline.scene_linear(raw, settings, sensor_codes=True),
                               np.clip(codes @ np.array(matrix, np.float32).reshape(3, 3).T, 0, 1), rtol=1e-6)
    # Lightings without a recorded matrix keep the balanced samples unchanged.
    np.testing.assert_allclose(pipeline.scene_linear(raw, settings.copy(lighting="Tungsten")), balanced)


def test_pro_slr_erimm_route_tones_before_unique_matrix(monkeypatch):
    from types import SimpleNamespace
    deviant = (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    unique = (1.25, -0.25, 0.0, 0.0, 1.0, 0.0, -0.5, 0.0, 1.5)
    raw = SimpleNamespace(camera_rgb=np.array([[[0.4, 0.2, 0.8]]], np.float32),
                          caps=camera.capabilities("DCS Pro SLR/n"), white_level=4095.0,
                          color_matrices={"Daylight": deviant}, unique_matrices={"Daylight": unique})
    monkeypatch.setattr(pipeline, "white_balance", lambda *_: np.array([1.0, 1.0, 1.5]))
    settings = pipeline.Settings(lighting="Daylight", tone=(-8.0, 0.5, 3.0))
    balanced = np.clip(raw.camera_rgb * np.float32([1.0, 1.0, 1.5]), 0, 1)
    toned = pipeline.pro_slr_tone_lut(settings.tone)[np.floor(balanced * 4095 + 0.5).astype(np.uint16)]
    codes = np.clip(toned @ np.array(unique, np.float32).reshape(3, 3).T, 0, 1)
    expected = pipeline.PRO_SLR_ERIMM_TABLE[np.floor(codes * 4095 + 0.5).astype(np.uint16)]
    np.testing.assert_array_equal(pipeline.pro_slr_erimm(raw, settings), expected)
    assert pipeline.pro_slr_erimm(raw, settings.copy(lighting="Tungsten")) is None


def test_pro_slr_erimm_table_follows_native_erimm_encoding():
    table = np.round(pipeline.PRO_SLR_ERIMM_TABLE * 4095).astype(int)
    # Entries read from DCSPro4SLR.dll .data 0x66035bd8; the analytic form gives 116 where the DLL
    # stores 115 at code 2, its only differing entry.
    assert table[[0, 1, 3, 16, 256, 1024, 2048, 4095]].tolist() == [0, 58, 173, 663, 1560, 2008, 2232, 2456]
    assert table[2] - 115 == 1
    assert np.all(np.diff(table) >= 0)


@need_installer
def test_rgb_render_matches_optional_native_srgb_samples(profile_dir):
    import cv2
    import tifffile

    cases = (("srgb_ev0.tif", {}), ("srgb_ev1.tif", {"exposure": 1.0}))
    references = [(ROOT / "raw/sample_tiff" / name, fields) for name, fields in cases]
    if not SLR_N_SAMPLE.is_file() or not all(path.is_file() for path, _ in references):
        pytest.skip("Pro SLR/n sample RAW and the sRGB TIFF references are required")
    raw = pipeline.decode(SLR_N_SAMPLE, half_size=True)
    settings = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000,
                                 white_point_ref=5000, tone=(-9.5, -0.2, 4.0), sharpen_amount=0)
    for reference, fields in references:
        # Compared at a quarter of full size, which averages out most of the remaining
        # half-size binning versus Kodak demosaic differences; measured RMSE is about .0089-.0091.
        size = (raw.camera_rgb.shape[1] // 2, raw.camera_rgb.shape[0] // 2)
        source = tifffile.imread(reference)
        expected = cv2.resize(source.astype(np.float32) / 65535.0, size, interpolation=cv2.INTER_AREA)
        image = pipeline.render(raw, settings.copy(**fields), profile_directory=profile_dir, quality="faster")
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        assert rmse < 0.0095, f"{reference.name} sRGB RMSE {rmse:.5f}"


@need_installer
def test_full_size_rgb_render_matches_optional_native_srgb_sample(profile_dir):
    import tifffile

    # srgb_ev0 was exported at Lens Optimization 0, the GQ432034/exposure references at the
    # camera's 10; measured RMSE .0046, .0079 and .0084 with the native Omen blue gain map
    cases = (("srgb_ev0.tif", {"lens_optimization": 0}, 0.0052),
             ("GQ432034.tif", {"lens_optimization": 10}, 0.0088),
             ("exposure_plus1.tif", {"lens_optimization": 10, "exposure": 1.0}, 0.0092))
    references = [(ROOT / "raw/sample_tiff" / name, fields, limit) for name, fields, limit in cases]
    if not SLR_N_SAMPLE.is_file() or not all(path.is_file() for path, _, _ in references):
        pytest.skip("Pro SLR/n sample RAW and the sRGB TIFF references are required")
    raw = pipeline.decode(SLR_N_SAMPLE)
    settings = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000,
                                 white_point_ref=5000, tone=(-9.5, -0.2, 4.0), sharpen_amount=0)
    for reference, fields, limit in references:
        image = pipeline.render(raw, settings.copy(**fields), profile_directory=profile_dir, quality="faster")
        expected = tifffile.imread(reference).astype(np.float32) / 65535.0
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        assert rmse < limit, f"{reference.name} full-size sRGB RMSE {rmse:.5f}"


@need_installer
def test_full_size_rgb_render_matches_optional_native_balance_and_lighting_samples(profile_dir):
    import tifffile

    folder = ROOT / "raw/sample_tiff/test4"
    names = ("밸런스-자동.tif", "클릭.tif", "형광등.tif", "플래쉬.tif")
    if not SLR_N_SAMPLE.is_file() or not all((folder / name).is_file() for name in names):
        pytest.skip("Pro SLR/n sample RAW and the balance/lighting sRGB references are required")
    raw = pipeline.decode(SLR_N_SAMPLE)
    base = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000, white_point_ref=5000,
                             tone=(-9.5, -0.2, 4.0), sharpen_amount=0, lens_optimization=0)
    # Click at image pixel (2188, 2078), as shown in Photo Desk's status bar.
    click = pipeline.click_wb_pro_slr(pipeline.read_cfa(SLR_N_SAMPLE), 2188, 2078, 0)
    flash = base.copy(lighting="Flash")
    flash_temperature = pipeline.balance_temperature(raw, flash)   # the Flash curve's nominal 5500 K
    cases = (
        (names[0], base.copy(balance="auto", kelvin_mode=5, kelvin_illumination="Daylight"), 0.0052),
        (names[1], base.copy(balance="click", click_wb=click), 0.0052),
        (names[2], base.copy(lighting="Fluorescent"), 0.0053),
        (names[3], flash.copy(white_point=flash_temperature, white_point_ref=flash_temperature), 0.0053),
    )
    for name, settings, limit in cases:
        image = pipeline.render(raw, settings, profile_directory=profile_dir, quality="faster")
        expected = tifffile.imread(folder / name).astype(np.float32) / 65535.0
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        assert rmse < limit, f"{name} full-size sRGB RMSE {rmse:.5f}"


def test_full_size_linear_render_matches_optional_native_samples():
    import tifffile

    folder = ROOT / "raw/sample_tiff"
    cases = (("linear_ev0.tif", {}), ("linear_ev1.tif", {"exposure": 1.0}),
             ("test1/plus2.0.tif", {"exposure": 2.0}),
             ("test1/텅스텐.tif", {"lighting": "Tungsten", "white_point": 3200, "white_point_ref": 3200}),
             ("test1/7500K.tif", {"white_point": 7500, "white_point_ref": 7500}))
    if not SLR_N_SAMPLE.is_file() or not all((folder / name).is_file() for name, _ in cases):
        pytest.skip("Pro SLR/n sample RAW and the linear TIFF references are required")
    raw = pipeline.decode(SLR_N_SAMPLE)
    settings = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5000,
                                 white_point_ref=5000, tone=(-9.5, -0.2, 4.0), sharpen_amount=0)
    for name, fields in cases:
        expected = tifffile.imread(folder / name).astype(np.float32) / 65535.0
        image = pipeline.render(raw, settings.copy(**fields), quality="faster", linear=True)
        rmse = float(np.sqrt(np.mean((image - expected) ** 2, dtype=np.float64)))
        # Pixel-for-pixel with the native AH2 demosaic: .0016-.0022 (LibRaw DHT gave .0026-.0039).
        assert rmse < 0.0026, f"{name} full-size linear RMSE {rmse:.5f}"


def test_kodak_balance_cfa_uses_fixed_point_gains_and_saturates():
    from photodesk import demosaic

    cfa = np.array([[1000, 1000], [1000, 3700]], np.uint16)   # G R / B G
    out = demosaic.balance_cfa(cfa, (1.23633, 1.0, 1.4297), 3700)
    # round(1.23633 * 1024) = 1266, round(1.4297 * 1024) = 1464
    assert out.tolist() == [[1000, (1000 * 1266 + 512) >> 10], [(1000 * 1464 + 512) >> 10, 3700]]
    assert demosaic.balance_cfa(np.array([[3000, 3000], [3000, 3000]]), (1.5, 1.0, 1.5), 3700).max() == 3700


def test_kodak_ah2_demosaic_reproduces_flat_fields_and_keeps_site_samples():
    from photodesk import demosaic

    flat = np.zeros((16, 20), np.int32)
    flat[0::2, 0::2], flat[0::2, 1::2], flat[1::2, 0::2], flat[1::2, 1::2] = 800, 1200, 600, 800
    rgb = demosaic.ah2(flat)
    assert rgb.dtype == np.uint16 and rgb.shape == (16, 20, 3)
    assert np.all(rgb[..., 0] == 1200) and np.all(rgb[..., 1] == 800) and np.all(rgb[..., 2] == 600)
    # A vertical edge: green at red/blue sites is interpolated along the edge, never across it.
    edge = np.where(np.arange(20)[None, :] < 10, 400, 2400).repeat(16, 0).astype(np.int32)
    rgb = demosaic.ah2(edge, threshold=0)
    assert np.all(rgb[2:-2, :9, 1] == 400) and np.all(rgb[2:-2, 11:, 1] == 2400)
    # Banded processing matches a single pass.
    noise = np.random.default_rng(1).integers(0, 4096, (40, 24)).astype(np.int32)
    np.testing.assert_array_equal(demosaic.ah2(noise, band=8), demosaic.ah2(noise, band=64))


def test_color_matrices_read_from_slrn_sample():
    if not SLR_N_SAMPLE.is_file():
        pytest.skip("SLR/n sample not available")
    matrices = camera.color_matrices(SLR_N_SAMPLE)
    assert set(matrices) == set(camera.ILLUMINANTS)
    np.testing.assert_allclose(matrices["Daylight"], (1.046875, -0.03515625, -0.01171875, -0.0234375,
                                                      1.1640625, -0.140625, -0.06640625, 0.19140625, 0.875))
    for values in matrices.values():
        np.testing.assert_allclose(np.reshape(values, (3, 3)).sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(not SLR_N_SAMPLE.is_file(), reason="Pro SLR/n sample RAW not available")
@need_installer
def test_pro_slrn_native_lens_optimization_is_identity_at_the_camera_strength(profile_dir):
    raw = pipeline.decode(SLR_N_SAMPLE, half_size=True)
    assert raw.lens_omen["baseline"] == 10
    settings = pipeline.Settings(sharpen_amount=0)
    unadjusted = pipeline.render(raw, settings.copy(lens_optimization=0), profile_directory=profile_dir)
    matching = pipeline.render(raw, settings.copy(lens_optimization=10), profile_directory=profile_dir)
    stronger = pipeline.render(raw, settings.copy(lens_optimization=40), profile_directory=profile_dir)
    # strength 10 is the camera's own: no Omen gain and no Lab heuristic; any other strength
    # changes the blue falloff only away from the centre
    h, w = unadjusted.shape[:2]
    centre = (slice(h // 2 - 8, h // 2 + 8), slice(w // 2 - 8, w // 2 + 8))
    edge = (slice(None), slice(0, w // 16))
    assert not np.allclose(matching, unadjusted)
    assert np.abs(stronger[centre] - matching[centre]).mean() < 2e-3
    assert np.abs(stronger[edge][..., 2] - matching[edge][..., 2]).mean() > 5e-3


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
    assert page.dtype == np.uint16
    assert page.shape in ((4072, 4072, 3), (4080, 4080, 3))
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


@pytest.mark.parametrize("tiff_16bit,scale,dtype", [(False, 255, np.uint8), (True, 65535, np.uint16)])
def test_tiff_upscale_cubic_overshoot_saturates(tmp_path, tiff_16bit, scale, dtype):
    import tifffile

    pattern = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], np.float32)
    img = np.repeat(pattern[..., None], 3, axis=2)
    opts = export.OutputOptions(resize=True, percent=400, tiff_16bit=tiff_16bit)
    resized = export.apply_size(img, opts)
    assert resized.min() < 0 and resized.max() > 1

    saved = export.save(img, tmp_path / f"overshoot-{scale}.tif", "tiff", opts, {})
    actual = tifffile.imread(saved)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, np.round(np.clip(resized, 0, 1) * scale).astype(dtype))
    assert actual.min() == 0 and actual.max() == scale


def test_pro_slrn_linear_tiff_uses_provisional_12bit_lattice(tmp_path):
    import tifffile

    img = np.array([[[-0.1, 0.0, 0.0002], [0.25, 0.5, 0.75], [1.0, 1.1, 2.0]]],
                   dtype=np.float32)
    opts = export.OutputOptions(tiff_16bit=True, embed_tiff_profile=False)
    saved = export.save(img, tmp_path / "linear-lattice.tif", "tiff", opts, {},
                        _quantize_12bit_lattice=True)

    with tifffile.TiffFile(saved) as tiff:
        page = tiff.pages[0]
        actual = page.asarray()
        assert 34675 not in page.tags
    expected = np.array([[[0, 0, 16], [16384, 32768, 49136], [65520, 65520, 65520]]],
                        dtype=np.uint16)
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual % 16 == 0)
    assert actual.max() == 65520


def test_tiff_16bit_default_conversion_remains_full_range(tmp_path):
    import tifffile

    img = np.array([[[0.1, 0.5, 1.0]]], dtype=np.float32)
    opts = export.OutputOptions(tiff_16bit=True, embed_tiff_profile=False)
    saved = export.save(img, tmp_path / "ordinary-16bit.tif", "tiff", opts, {})
    actual = tifffile.imread(saved)

    np.testing.assert_array_equal(actual, np.round(img * 65535).astype(np.uint16))
    assert actual[0, 0, 0] % 16 != 0 and actual[0, 0, 2] == 65535


@pytest.mark.parametrize(
    "model,dcs,file_type,linear,bits16,resize,crop,gps,eligible",
    [
        ("DCS Pro SLR/n", True, "tiff", True, True, False, None, False, True),
        ("DCS Pro SLR/c", True, "tiff", True, True, False, None, False, False),
        ("DCS Pro 14n", True, "tiff", True, True, False, None, False, False),
        ("ProBack645", True, "tiff", True, True, False, None, False, False),
        ("Canon EOS 5D", False, "tiff", True, True, False, None, False, False),
        ("DCS Pro SLR/n", True, "tiff", False, True, False, None, False, False),
        ("DCS Pro SLR/n", True, "tiff", True, False, False, None, False, False),
        ("DCS Pro SLR/n", True, "jpeg", True, True, False, None, False, False),
        ("DCS Pro SLR/n", True, "tiff", True, True, True, None, False, False),
        ("DCS Pro SLR/n", True, "tiff", True, True, False, (0.1, 0.1, 0.9, 0.9), False, False),
        ("DCS Pro SLR/n", True, "tiff", True, True, False, (0, 0, 1, 1), False, False),
        ("DCS Pro SLR/n", True, "tiff", True, True, False, None, True, False),
    ],
)
def test_process_file_gates_pro_slrn_lattice_dispatch(
        tmp_path, monkeypatch, model, dcs, file_type, linear, bits16, resize, crop, gps, eligible):
    from types import SimpleNamespace

    image = np.zeros((2, 3, 3), dtype=np.float32)
    metadata = {"model": model}
    if gps:
        metadata["gps"] = {"lat": 1.0, "lon": 2.0}
    raw = SimpleNamespace(caps=SimpleNamespace(dcs=dcs), metadata=metadata)
    profile = tmp_path / "working.icc"
    profile.write_bytes(b"synthetic profile")
    rendered = []
    saves = []
    gps_overlays = []
    monkeypatch.setattr(export, "is_raw", lambda _: True)
    monkeypatch.setattr(pipeline, "decode", lambda _: raw)
    monkeypatch.setattr(pipeline, "resolve_space", lambda *_: str(profile))
    monkeypatch.setattr(pipeline, "render",
                        lambda *args, **kwargs: rendered.append(kwargs["linear"]) or image)
    monkeypatch.setattr(export, "render_gps",
                        lambda img, text: gps_overlays.append(text) or img)
    monkeypatch.setattr(camera, "gps_text", lambda _: "synthetic GPS")

    def capture_save(*args, **kwargs):
        saves.append(kwargs.get("_quantize_12bit_lattice", False))
        return Path(args[1])

    monkeypatch.setattr(export, "save", capture_save)
    opts = export.OutputOptions(tiff_linear=linear, tiff_16bit=bits16,
                                resize=resize, render_gps=gps)
    settings = pipeline.Settings(crop=crop)
    dest = tmp_path / "out"
    assert export.process_file("synthetic.dcr", dest, file_type, opts, settings, tmp_path) == dest

    assert saves == [eligible]
    assert rendered == [file_type == "tiff" and linear and dcs]
    assert bool(gps_overlays) is gps


def test_finished_file_bypasses_pro_slrn_lattice_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(export, "is_raw", lambda _: False)
    monkeypatch.setattr(export, "process_finished",
                        lambda *args: calls.append(args) or Path(args[1]))
    monkeypatch.setattr(pipeline, "decode", lambda *_: pytest.fail("finished file decoded as RAW"))
    opts = export.OutputOptions(tiff_linear=True, tiff_16bit=True)
    settings = pipeline.Settings()
    dest = tmp_path / "finished.tif"

    assert export.process_file("finished.jpg", dest, "tiff", opts, settings) == dest
    assert len(calls) == 1 and calls[0][0] == "finished.jpg" and calls[0][2] == "tiff"


def test_jpeg_upscale_cubic_overshoot_saturates(tmp_path):
    from PIL import Image

    pattern = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], np.float32)
    img = np.repeat(pattern[..., None], 3, axis=2)
    opts = export.OutputOptions(resize=True, percent=400, jpeg_quality="best")
    resized = export.apply_size(img, opts)
    assert resized.min() < 0 and resized.max() > 1
    peak = np.unravel_index(np.argmax(resized[..., 0]), resized.shape[:2])
    trough = np.unravel_index(np.argmin(resized[..., 0]), resized.shape[:2])

    saved = export.save(img, tmp_path / "overshoot.jpg", "jpeg", opts, {})
    with Image.open(saved) as jpeg:
        actual = np.asarray(jpeg.convert("RGB"))
    # JPEG is lossy, but clipped extrema remain near white and black. Unsigned wraparound
    # would instead turn these overshoots into dark and bright values, respectively.
    assert actual[peak].min() > 160
    assert actual[trough].max() < 95


@pytest.mark.parametrize("target_serial,source_serial,allowed", [
    ("camera-a", "camera-a", True), ("camera-b", "camera-a", False), (None, None, False),
])
def test_paste_click_balance_requires_matching_known_camera(target_serial, source_serial, allowed):
    from types import SimpleNamespace
    from photodesk.gui.model import PASTE_GROUPS, paste_into

    source = pipeline.Settings(balance="click", click_wb=(1.3, 1.0, 1.4), lighting="Tungsten",
                               white_point=4500, white_point_ref=4500, kelvin_mode=6,
                               kelvin_illumination="Tungsten", saved_balance_point=(2048, 2500, 1800),
                               color_balance=(3, 4, 5))
    destination = pipeline.Settings(balance="preset", lighting="Daylight", white_point=5500,
                                    white_point_ref=5500, kelvin_mode=0, kelvin_illumination="Daylight")
    before = destination.copy()
    paste_into(SimpleNamespace(is_raw=True, serial=target_serial), destination, source,
               PASTE_GROUPS[1344], source_serial)
    assert destination.color_balance == source.color_balance
    for field in ("balance", "click_wb", "lighting", "white_point", "white_point_ref", "kelvin_mode",
                  "kelvin_illumination", "saved_balance_point"):
        assert getattr(destination, field) == getattr(source if allowed else before, field)


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
