#!/usr/bin/env python3
"""Blend an explicit, ordered list of image files with a BlendStack mode.

Unlike ``blend_folder.py`` (which globs and sorts a directory), this takes
the exact files to blend, in the exact fold order given on the command line
— the first file is the base. That makes it the right entry point for
frontends that already know which images to combine and in what order, such
as the darktable Lua export integration (``darktable_plugin/blendstack.lua``).

Usage:

    blend_cli.py --mode canon_bright --out result.tif img1.tif img2.tif ...
    blend_cli.py --mode average --out mean.tif *.tif

All blend controls are optional; softness/bias/basis apply only to the Canon
comparative modes and are ignored by the continuous modes. Exit code 0 on
success (the output path is printed on the last line), non-zero on error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run straight from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blendstack.core import engine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blend an ordered list of images with a BlendStack mode "
        "(first file = base)."
    )
    parser.add_argument("images", type=Path, nargs="+", help="Input files, in fold order")
    parser.add_argument("--mode", choices=engine.mode_names(), default="canon_bright")
    parser.add_argument("--softness", type=float, default=0.0, help="0-100 (Canon modes)")
    parser.add_argument("--bias", type=float, default=0.0, help="-100..100 (Canon modes)")
    parser.add_argument(
        "--basis", choices=("per_channel", "luminance"), default="per_channel",
        help="Comparison basis (Canon modes)",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output file (.tif/.tiff = 16-bit, .png = 16-bit, .jpg = 8-bit)",
    )
    args = parser.parse_args(argv)

    paths = args.images
    if not engine.MIN_IMAGES <= len(paths) <= engine.MAX_IMAGES:
        parser.error(
            f"got {len(paths)} image(s); need "
            f"{engine.MIN_IMAGES}-{engine.MAX_IMAGES}"
        )
    missing = [p for p in paths if not p.is_file()]
    if missing:
        parser.error("missing input file(s): " + ", ".join(str(p) for p in missing))

    params = {"softness": args.softness, "bias": args.bias, "basis": args.basis}
    try:
        out = engine.blend_files(
            [str(p) for p in paths], mode=args.mode, params=params,
            out_path=str(args.out),
        )
    except Exception as exc:  # surface a clean one-line error to the caller
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Machine-readable success line last: the darktable script reads it.
    print(f"OK {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
