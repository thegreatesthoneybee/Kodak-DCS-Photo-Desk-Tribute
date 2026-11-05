"""Portable RGB choices survive relocation and embed the actual output profile."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from photodesk import export, icc, pipeline, profiles


@pytest.fixture
def portable_spaces(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "data_dir", lambda: tmp_path / "user-data")
    monkeypatch.setattr(profiles, "system_color_dirs_profiles", lambda: iter(()))
    directory = tmp_path / "read-only-bundle"
    directory.mkdir()
    spaces = pipeline.working_spaces(directory)
    return directory, spaces


def test_portable_working_spaces_are_real_profiles_with_relocatable_settings(portable_spaces):
    directory, spaces = portable_spaces
    assert {desc for _, desc in spaces} == {
        "Adobe RGB (1998)", "image-P3", "Kodak DCS RGB", "Kodak ProPhoto RGB",
        "Kodak ERIMM Color Space to Scene Reference Space", "KODAK sRGB Display", "sRGB IEC61966-2.1",
    }
    assert not list(directory.iterdir())
    assert not profiles.data_dir().exists()
    for path, _ in spaces:
        setting = pipeline.space_setting(path, directory)
        assert setting == Path(path).name
        assert pipeline.resolve_space(setting, directory) == path
        assert pipeline.resolve_space(str(directory / "moved" / setting), directory) == path
        assert icc.Profile(path).is_working_space
    # Discovery and lookup are read-only, even with no system/user profiles installed.
    mtimes = {path: Path(path).stat().st_mtime_ns for path, _ in spaces}
    assert pipeline.working_spaces(directory) == spaces
    assert all(Path(path).stat().st_mtime_ns == stamp for path, stamp in mtimes.items())


@pytest.mark.parametrize("file_type", ["jpeg", "tiff"])
def test_rgb_space_conversion_and_embedded_profile_roundtrip(portable_spaces, tmp_path, file_type):
    directory, spaces = portable_spaces
    xyz = np.array([[[0, 0, 0], icc.D50, icc.D50 * 0.18, [0.3, 0.2, 0.1]]], np.float32)
    outputs = []
    for i, (path, _) in enumerate(spaces):
        chosen = pipeline.resolve_space(pipeline.space_setting(path, directory), directory)
        profile = icc.Profile(chosen)
        rgb = profile.encoder()(xyz)
        assert rgb.shape == xyz.shape and np.isfinite(rgb).all()
        outputs.append(rgb)
        data = Path(chosen).read_bytes()
        target = tmp_path / f"output-{i}.{file_type}"
        opts = export.OutputOptions(working_space=Path(chosen).name, tiff_16bit=True)
        export.save(rgb, target, file_type, opts, {}, space_bytes=data)
        if file_type == "jpeg":
            with Image.open(target) as image:
                assert image.mode == "RGB"
                assert image.info["icc_profile"] == data
        else:
            with tifffile.TiffFile(target) as image:
                assert image.pages[0].samplesperpixel == 3
                assert image.pages[0].tags[34675].value == data
                np.testing.assert_array_equal(image.asarray(), np.round(rgb * 65535).astype(np.uint16))
    assert not np.allclose(outputs[0], outputs[1], atol=1e-3)
