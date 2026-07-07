"""PyInstaller entry point for the BlendStack macOS app bundle.

Kept deliberately tiny: all real logic lives in :mod:`blendstack.app`.
PyInstaller analyses this script, follows the imports, and freezes the
whole ``blendstack`` package (core + app) into the bundle.
"""

from __future__ import annotations

import sys

from blendstack.app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
