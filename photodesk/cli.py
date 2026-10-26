"""Command line interface: profile installation and batch conversion."""
import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

from . import export, pipeline, profiles

RAW_SUFFIXES = {".dcr", ".kdc", ".dc2", ".k25", ".tif"}
FILE_TYPES = {"jpeg": "jpeg", "jpg": "jpeg", "tiff": "tiff", "tif": "tiff"}


def collect(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.iterdir() if f.suffix.lower() in RAW_SUFFIXES and f.is_file())
        else:
            yield p


def _convert(job):
    src, args = job
    settings = pipeline.load_settings(src)
    if args.look:
        settings.look = args.look
    if args.lighting:
        settings.lighting = args.lighting
    if args.balance:
        settings.balance = args.balance
    opts = export.OutputOptions(working_space=args.space, tiff_16bit=args.bits == 16,
                                tiff_linear=args.linear, jpeg_quality=args.quality)
    if args.long_edge:
        opts.resize, opts.scale_style, opts.width, opts.height = True, "fit_trim", args.long_edge, args.long_edge
    file_type = FILE_TYPES[args.format]
    dest = export.output_path(src, args.out, file_type)
    return export.process_file(src, dest, file_type, opts, settings)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="photodesk")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ip = sub.add_parser("install-profiles", help="extract Kodak profiles and UI resources from Photo_Desk_V_4_3_0.exe")
    ip.add_argument("installer")
    ip.add_argument("--dest")

    cp = sub.add_parser("convert", help="convert raw files (Standard TIFF / Standard JPEG)")
    cp.add_argument("inputs", nargs="+")
    cp.add_argument("-o", "--out", required=True)
    cp.add_argument("-f", "--format", choices=sorted(FILE_TYPES), default="jpeg")
    cp.add_argument("-s", "--space", default=pipeline.DEFAULT_WORKING_SPACE,
                    help="RGB working space profile (file name in the profile folder or a path)")
    cp.add_argument("-q", "--quality", choices=list(export.JPEG_QUALITY), default="best")
    cp.add_argument("--bits", type=int, choices=(8, 16), default=8)
    cp.add_argument("--linear", action="store_true", help="TIFF Linear Mode")
    cp.add_argument("--long-edge", type=int)
    cp.add_argument("--look", help="look profile file name, e.g. DCSPortraitLook.icm")
    cp.add_argument("--lighting", choices=pipeline.LIGHTINGS)
    cp.add_argument("--balance", choices=["preset", "custom", "auto"])
    # Each render already uses every core; a few files in flight overlap the single-threaded raw
    # decoding and file encoding (8 files: 1 job 23 s, 2 jobs 16 s, 4 jobs 13 s on 20 threads).
    cp.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1))

    sub.add_parser("looks", help="list installed look profiles")

    args = ap.parse_args(argv)
    if args.cmd == "install-profiles":
        dest, names = profiles.install_from_installer(args.installer, args.dest)
        print(f"installed {len(names)} profiles to {dest}")
        return 0
    if args.cmd == "looks":
        for name, desc in pipeline.available_looks():
            print(f"{name}\t{desc}")
        return 0

    if not profiles.is_installed():
        print("Kodak profiles not installed; run: photodesk install-profiles Photo_Desk_V_4_3_0.exe",
              file=sys.stderr)
        return 2
    files = list(collect(args.inputs))
    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_convert, (f, args)): f for f in files}
        for fut in concurrent.futures.as_completed(futures):
            try:
                print(f"{futures[fut]} -> {fut.result()}")
            except Exception as e:  # report and continue with remaining files
                failed += 1
                print(f"{futures[fut]}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
