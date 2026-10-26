"""Timing of the main processing paths (not a test): python tests/bench.py <raw file> [profile dir]

Prints seconds for decode, colour LUT build, thumbnail/preview/full renders and export.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photodesk import export, pipeline, profiles  # noqa: E402


def timed(label, fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    print(f"{label:<34}{time.perf_counter() - t:8.3f} s")
    return out


def main():
    raw_path = Path(sys.argv[1])
    pdir = str(sys.argv[2] if len(sys.argv) > 2 else profiles.profile_dir())
    s = pipeline.Settings()
    raw = timed("decode (full)", pipeline.decode, raw_path)
    timed("decode (half size)", pipeline.decode, raw_path, half_size=True)
    timed("embedded thumbnail", pipeline.thumbnail, raw_path)
    space = pipeline.resolve_space(pipeline.DISPLAY_SPACE, pdir)
    source = pipeline.source_profile_for(raw, s, pdir)
    pipeline.color_lut.cache_clear()
    timed("colour LUT build", pipeline.color_lut, source, s.look, space, pdir)
    small = pipeline.reduced(raw, 256)
    timed("thumbnail render (256)", pipeline.render, small, s, profile_directory=pdir, quality="faster")
    prev = pipeline.reduced(raw, 1600)
    timed("preview render faster (1600)", pipeline.render, prev, s, profile_directory=pdir, quality="faster")
    sharp = s.copy(sharpen_amount=6.0, noise_type="advanced", noise_radius=3, noise_strength=50)
    timed("preview render better (1600)", pipeline.render, prev, sharp, profile_directory=pdir)
    timed("full render (no finish)", pipeline.render, raw, s, profile_directory=pdir, quality="faster")
    timed("full render + NR/sharpen", pipeline.render, raw, sharp, profile_directory=pdir)
    out = Path(tempfile.mkdtemp())
    timed("export JPEG (full)", export.process_file, raw_path, out / "a.jpg", "jpeg",
          export.OutputOptions(), s, pdir)
    timed("export TIFF 16-bit (full)", export.process_file, raw_path, out / "a.tif", "tiff",
          export.OutputOptions(tiff_16bit=True), s, pdir)


if __name__ == "__main__":
    main()
