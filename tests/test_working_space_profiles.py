import io

import numpy as np
from PIL import Image, ImageCms

from photodesk import icc
from photodesk.working_space_profiles import builtin_profiles


# D50-adapted XYZ colorants calculated independently from each standard's published
# primaries and D65 white using Bradford adaptation, then rounded to 6 decimals.
_REFERENCE_COLORANTS = {
    "AdobeRGB1998.icc": np.array([
        [0.609774, 0.205249, 0.149177],
        [0.311133, 0.625662, 0.063205],
        [0.019472, 0.060880, 0.744548],
    ]),
    "image-P3.icc": np.array([
        [0.515157, 0.291948, 0.157095],
        [0.241208, 0.692232, 0.066560],
        [-0.001050, 0.041881, 0.784069],
    ]),
}


def _xyz_to_lab(xyz):
    t = xyz / np.array([0.9642, 1.0, 0.8249])
    delta = 6 / 29
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def test_bundled_wide_gamut_profiles_match_standard_matrix_and_trc(tmp_path):
    profiles = builtin_profiles()
    assert profiles == builtin_profiles()
    assert {"AdobeRGB1998.icc", "image-P3.icc"} <= set(profiles)

    descriptions = {
        "AdobeRGB1998.icc": "Adobe RGB (1998)",
        "image-P3.icc": "image-P3",
    }
    for filename in descriptions:
        data = profiles[filename]
        path = tmp_path / filename
        path.write_bytes(data)
        profile = icc.Profile(path)
        assert profile.description == descriptions[filename]
        assert profile.is_working_space
        assert set(("desc", "wtpt", "rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC", "cprt")) <= set(profile.tags)
        np.testing.assert_allclose(profile.rgb_to_xyz_matrix(), _REFERENCE_COLORANTS[filename], atol=2e-5)
        copyright_text = ImageCms.getProfileCopyright(ImageCms.ImageCmsProfile(io.BytesIO(data)))
        assert "Independently generated" in copyright_text
        assert "not a vendor binary" in copyright_text

    adobe = icc.Profile(_write_profile(tmp_path, "AdobeRGB1998.icc", profiles["AdobeRGB1998.icc"]))
    x, y = adobe.trc("r")
    np.testing.assert_allclose(np.interp(0.5, x, y), 0.5 ** (563 / 256), atol=2e-4)

    p3 = icc.Profile(_write_profile(tmp_path, "image-P3.icc", profiles["image-P3.icc"]))
    x, y = p3.trc("g")
    expected_mid = 0.5 / 12.92 if 0.5 <= 0.04045 else ((0.5 + 0.055) / 1.055) ** 2.4
    np.testing.assert_allclose(np.interp(0.5, x, y), expected_mid, atol=2e-4)


def _write_profile(directory, filename, data):
    path = directory / filename
    path.write_bytes(data)
    return path


def test_pillow_loads_profiles_and_converts_neutral_and_primary_colors():
    for filename in _REFERENCE_COLORANTS:
        data = builtin_profiles()[filename]
        source = ImageCms.ImageCmsProfile(io.BytesIO(data))
        assert ImageCms.getProfileDescription(source).strip() == (
            "Adobe RGB (1998)" if filename.startswith("Adobe") else "image-P3"
        )

        rgb_samples = Image.frombytes("RGB", (3, 1), bytes((0, 0, 0, 255, 255, 255, 128, 128, 128)))
        to_srgb = ImageCms.buildTransform(source, ImageCms.createProfile("sRGB"), "RGB", "RGB",
                                          renderingIntent=0)
        converted = np.asarray(ImageCms.applyTransform(rgb_samples, to_srgb))[0]
        np.testing.assert_allclose(converted[0], (0, 0, 0), atol=1)
        np.testing.assert_allclose(converted[1], (255, 255, 255), atol=1)
        np.testing.assert_allclose(converted[2], (128, 128, 128), atol=2)

        primary_samples = Image.frombytes("RGB", (3, 1), bytes((255, 0, 0, 0, 255, 0, 0, 0, 255)))
        to_lab = ImageCms.buildTransform(source, ImageCms.createProfile("LAB"), "RGB", "LAB",
                                         renderingIntent=0)
        lab_image = ImageCms.applyTransform(primary_samples, to_lab)
        # Pillow's LAB array interface stores signed a/b bytes, while getpixel
        # returns the conventional 128-offset values used below.
        lab_bytes = np.array([lab_image.getpixel((i, 0)) for i in range(3)], np.float64)
        actual_lab = np.stack((lab_bytes[:, 0] * (100 / 255),
                               lab_bytes[:, 1] - 128,
                               lab_bytes[:, 2] - 128), axis=-1)
        reference_lab = _xyz_to_lab(_REFERENCE_COLORANTS[filename].T)
        np.testing.assert_allclose(actual_lab, reference_lab, atol=2.5)
