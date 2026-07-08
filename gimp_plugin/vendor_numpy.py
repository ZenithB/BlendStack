#!/usr/bin/env python3
"""Vendor a NumPy wheel into the GIMP plugin folder (brief §6, mitigation 2).

GIMP's bundled Python on macOS ships without NumPy, and — on current
Apple Silicon macOS — the documented one-line fix (``... python3 -m pip
install numpy`` run against GIMP's own interpreter) does not work: macOS
Launch Constraints refuse to let GIMP's embedded ``python3.10`` be launched
directly from a shell at all (AMFI "Launch Constraint Violation", enforced
at the kernel level; this is a hard OS policy, not something to bypass by
weakening system security). So this script implements the brief's fallback
instead: download NumPy's official wheel for GIMP's exact interpreter
(cp310, macOS arm64) and vendor its unpacked contents directly into the
plugin folder, where the plugin's own sys.path insertion (see
blendstack-blend.py) picks it up with no pip, no network, and no directly
executing GIMP's constrained interpreter.

Re-run after bumping the target NumPy version:

    python3 gimp_plugin/vendor_numpy.py [numpy-version]

Test-only files (``**/tests/``) are excluded to keep the vendored copy
smaller; nothing else is trimmed, to avoid subtly breaking NumPy internals.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "gimp_plugin" / "blendstack-blend" / "vendor"

# GIMP 3.2's bundled macOS interpreter (verified via `file`/`codesign` on the
# actual app bundle) is a thin arm64 build of CPython 3.10.
PLATFORM = "macosx_11_0_arm64"
PY_VERSION = "310"
IMPLEMENTATION = "cp"
ABI = "cp310"

DEFAULT_NUMPY_VERSION = "numpy"  # unpinned = latest compatible


def main() -> None:
    spec = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NUMPY_VERSION

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download", spec,
                "--no-deps", "--only-binary=:all:",
                "--platform", PLATFORM,
                "--python-version", PY_VERSION,
                "--implementation", IMPLEMENTATION,
                "--abi", ABI,
                "-d", str(tmp_path),
            ],
            check=True,
        )
        wheels = list(tmp_path.glob("numpy-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"Expected exactly one wheel, got {wheels}")
        wheel = wheels[0]

        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(extract_dir)

        if DEST.exists():
            shutil.rmtree(DEST)
        DEST.mkdir(parents=True)

        ignore = shutil.ignore_patterns("tests", "__pycache__", "*.pyc")
        shutil.copytree(extract_dir / "numpy", DEST / "numpy", ignore=ignore)

        dist_info = next(extract_dir.glob("numpy-*.dist-info"))
        shutil.copytree(dist_info, DEST / dist_info.name)

        size_mb = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e6
        print(f"Vendored {wheel.name} -> {DEST} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
