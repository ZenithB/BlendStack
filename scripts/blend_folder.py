#!/usr/bin/env python3
"""Test harness CLI (project brief §7, Phase 1 deliverable).

Blends every supported image in a folder (sorted by filename) with the
chosen mode and writes the composite to disk.

Usage::

    python scripts/blend_folder.py PHOTOS_DIR --mode canon_bright \
        --softness 15 --bias -20 --basis luminance --out result.tif

Without ``--out`` the output lands in the input folder as
``blend_<mode>_<YYYYMMDD-HHMMSS>.tif``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blendstack.core import engine  # noqa: E402
from blendstack.core import io as bs_io  # noqa: E402


def _collect_images(folder: Path) -> list[Path]:
    """All supported image files in ``folder``, sorted by name."""
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in bs_io.SUPPORTED_INPUT_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blend all images in a folder with a Canon comparative mode."
    )
    parser.add_argument("folder", type=Path, help="Folder containing the input images")
    parser.add_argument(
        "--mode",
        choices=engine.mode_names(),
        default="canon_bright",
        help="Blend mode (default: canon_bright)",
    )
    parser.add_argument(
        "--softness", type=float, default=0.0,
        help="Selection softness 0-100 (0 = hard, pixel-exact Canon; default 0)",
    )
    parser.add_argument(
        "--bias", type=float, default=0.0,
        help="Comparison bias -100..+100, positive favours later images (default 0)",
    )
    parser.add_argument(
        "--basis", choices=("per_channel", "luminance"), default="per_channel",
        help="Comparison basis (default: per_channel = Canon-faithful)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output file (.tif/.png/.jpg). Default: blend_<mode>_<timestamp>.tif "
             "inside the input folder.",
    )
    args = parser.parse_args(argv)

    folder: Path = args.folder
    if not folder.is_dir():
        parser.error(f"not a folder: {folder}")
    paths = _collect_images(folder)
    if not engine.MIN_IMAGES <= len(paths) <= engine.MAX_IMAGES:
        parser.error(
            f"found {len(paths)} supported image(s) in {folder}; "
            f"need {engine.MIN_IMAGES}-{engine.MAX_IMAGES}"
        )

    print(f"Blending {len(paths)} images ({args.mode}):")
    for p in paths:
        print(f"  {p.name}")

    out = engine.blend_files(
        paths,
        mode=args.mode,
        params={"softness": args.softness, "bias": args.bias, "basis": args.basis},
        out_path=args.out,
        out_dir=folder,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
