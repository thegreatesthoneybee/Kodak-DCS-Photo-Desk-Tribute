import itertools
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from photodesk import icc, profiles


def _identity_lut16():
    fixed_identity = (65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    tag = bytearray(struct.pack(">4s4s4B9iHH", b"mft2", b"\0" * 4, 3, 3, 2, 0,
                                *fixed_identity, 2, 2))
    identity_curve = (0, 65535)
    for _ in range(3):
        tag.extend(struct.pack(">2H", *identity_curve))
    for node in itertools.product(identity_curve, repeat=3):
        tag.extend(struct.pack(">3H", *node))
    for _ in range(3):
        tag.extend(struct.pack(">2H", *identity_curve))
    return bytes(tag)


def _xyz_tag(xyz):
    return struct.pack(">4sI3i", b"XYZ ", 0, *(round(v * 65536) for v in xyz))


def _matrix_tags():
    # Invertible identity matrix and linear TRCs; the values are deliberately unlike D50.
    tags = {name: _xyz_tag(xyz) for name, xyz in zip(
        ("rXYZ", "gXYZ", "bXYZ"), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))}
    for channel in "rgb":
        tags[channel + "TRC"] = struct.pack(">4sII", b"curv", 0, 0)
    return tags


def _write_profile(path, device_class="mntr", color_space="RGB ", pcs="XYZ ", tags=None):
    tags = dict(tags or {})
    header = bytearray(128)
    header[12:16] = device_class.encode("ascii")
    header[16:20] = color_space.encode("ascii")
    header[20:24] = pcs.encode("ascii")
    header[36:40] = b"acsp"
    table_end = 132 + 12 * len(tags)
    profile = bytearray(header + struct.pack(">I", len(tags)) + b"\0" * (12 * len(tags)))
    offset = table_end
    for index, (signature, value) in enumerate(tags.items()):
        struct.pack_into(">4sII", profile, 132 + 12 * index, signature.encode("ascii"), offset, len(value))
        offset += len(value)
    for value in tags.values():
        profile.extend(value)
    struct.pack_into(">I", profile, 0, len(profile))
    path.write_bytes(profile)
    return path


def test_xyz_and_lab_b2a_encoders_use_their_icc_pcs_encodings(tmp_path):
    lut = _identity_lut16()
    xyz_path = _write_profile(tmp_path / "space.icm", device_class="spac", pcs="XYZ ",
                              tags={"B2A0": lut})
    lab_tags = {**_matrix_tags(), "B2A0": lut}
    lab_path = _write_profile(tmp_path / "monitor.icm", pcs="Lab ", tags=lab_tags)
    xyz_profile, lab_profile = icc.Profile(xyz_path), icc.Profile(lab_path)
    assert xyz_profile.is_working_space and lab_profile.is_working_space

    sample = np.array([[0, 0, 0], icc.D50, [0.2, 0.35, 0.5]], np.float32).reshape(1, 3, 3)
    xyz_rgb = xyz_profile.encoder()(sample)
    lab_rgb = lab_profile.encoder()(sample)
    assert xyz_rgb.shape == lab_rgb.shape == sample.shape
    np.testing.assert_allclose(xyz_rgb, icc.xyz_encode(sample), atol=2e-7)
    np.testing.assert_allclose(lab_rgb, icc.lab_encode(icc.xyz_to_lab(sample)), atol=2e-7)
    np.testing.assert_allclose(xyz_rgb[0, 0], 0, atol=2e-7)
    np.testing.assert_allclose(xyz_rgb[0, 1], icc.xyz_encode(icc.D50), atol=2e-7)
    np.testing.assert_allclose(lab_rgb[0, 0], icc.lab_encode([0, 0, 0]), atol=2e-7)
    np.testing.assert_allclose(lab_rgb[0, 1], icc.lab_encode([100, 0, 0]), atol=2e-7)


def test_matrix_trc_encoder_remains_available_without_lut(tmp_path):
    profile = icc.Profile(_write_profile(tmp_path / "matrix.icm", tags=_matrix_tags()))
    assert profile.is_working_space
    encode = profile.encoder()
    np.testing.assert_allclose(encode(icc.D50[None]), icc.D50[None], atol=2e-6)
    np.testing.assert_allclose(encode(np.zeros((1, 3), np.float32)), 0, atol=1e-7)


def test_unsupported_or_malformed_profiles_are_not_working_spaces(tmp_path):
    lut = _identity_lut16()
    malformed = {**_matrix_tags(), "B2A0": b"mft2"}
    profiles_to_check = [
        _write_profile(tmp_path / "bad-lut.icm", pcs="Lab ", tags=malformed),
        _write_profile(tmp_path / "missing-transform.icm", device_class="spac", tags={}),
        _write_profile(tmp_path / "lab-device.icm", color_space="Lab ", pcs="XYZ ",
                       tags={"B2A0": lut}),
        _write_profile(tmp_path / "cmyk.icm", color_space="CMYK", pcs="Lab ",
                       tags={"B2A0": lut}),
        _write_profile(tmp_path / "unsupported-pcs.icm", pcs="Yxy ", tags={"B2A0": lut}),
    ]
    for path in profiles_to_check:
        profile = icc.Profile(path)
        assert not profile.is_working_space
        with pytest.raises(ValueError, match="supported RGB working-space"):
            profile.encoder()


def _optional_profile_path(filename):
    candidates = []
    configured = os.environ.get("PHOTODESK_PROFILES")
    if configured:
        candidates.append(Path(configured) / filename)
    candidates.append(profiles.profile_dir() / filename)
    for path in candidates:
        if path.is_file():
            return path
    for path in profiles.system_color_dirs_profiles():
        if path.name.casefold() == filename.casefold():
            return path
    return None


@pytest.mark.parametrize(("filename", "pcs"), [("KODAK_sRGB.icm", "Lab"),
                                                ("ERIMM_PCS3.icm", "XYZ")])
def test_optional_kodak_rgb_lut_profiles_are_supported(filename, pcs):
    path = _optional_profile_path(filename)
    if path is None:
        pytest.skip(f"optional {filename} profile is not installed")
    profile = icc.Profile(path)
    assert profile.device_class in ("mntr", "spac")
    assert profile.color_space == "RGB" and profile.pcs == pcs
    assert profile.is_working_space

    xyz = np.array([[0, 0, 0], icc.D50, [0.2, 0.35, 0.5]], np.float32)
    if pcs == "XYZ":
        pcs_input = icc.xyz_encode(xyz)
    else:
        pcs_input = icc.lab_encode(icc.xyz_to_lab(xyz))
    expected = profile.lut("B2A0")(pcs_input)
    encoded = profile.encoder()(xyz)
    assert encoded.shape == (3, 3)
    np.testing.assert_allclose(encoded, expected, atol=1e-7)
    assert np.isfinite(encoded).all() and np.all((encoded >= 0) & (encoded <= 1))
