#!/usr/bin/env python3
"""Vendor the frozen BlendStack core into the GIMP plugin folder.

GIMP runs plugins with its own bundled Python interpreter, which cannot see
this repository, so the plugin bundles its own copy of the (frozen, tested)
core package.  This script is the single source of that copy — never edit
the vendored files by hand; re-run this script after any core change:

    python3 gimp_plugin/sync_core.py

Copies  blendstack/__init__.py  and  blendstack/core/  (excluding
__pycache__) into  gimp_plugin/blendstack-blend/blendstack/ .  The app/
package (PySide6 GUI) is deliberately NOT copied — the plugin only needs
the engine.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PKG = REPO_ROOT / "blendstack"
DEST_PKG = REPO_ROOT / "gimp_plugin" / "blendstack-blend" / "blendstack"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def main() -> None:
    if not (SRC_PKG / "core").is_dir():
        raise SystemExit(f"Cannot find core package at {SRC_PKG / 'core'}")

    if DEST_PKG.exists():
        shutil.rmtree(DEST_PKG)
    DEST_PKG.mkdir(parents=True)

    shutil.copy2(SRC_PKG / "__init__.py", DEST_PKG / "__init__.py")
    shutil.copytree(SRC_PKG / "core", DEST_PKG / "core", ignore=IGNORE)

    copied = sorted(p.relative_to(DEST_PKG) for p in DEST_PKG.rglob("*.py"))
    print(f"Vendored {len(copied)} files into {DEST_PKG}:")
    for rel in copied:
        print(f"  blendstack/{rel}")


if __name__ == "__main__":
    main()
