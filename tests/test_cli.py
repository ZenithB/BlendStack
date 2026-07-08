"""Smoke test for the Phase-1 test harness CLI (scripts/blend_folder.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "blend_folder", REPO_ROOT / "scripts" / "blend_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["blend_folder"] = module
    spec.loader.exec_module(module)
    return module


def _load_blend_cli():
    spec = importlib.util.spec_from_file_location(
        "blend_cli", REPO_ROOT / "scripts" / "blend_cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["blend_cli"] = module
    spec.loader.exec_module(module)
    return module


def test_blend_cli_explicit_list_and_order(tmp_path: Path, capsys) -> None:
    """scripts/blend_cli.py blends an explicit ordered file list (used by
    the darktable integration)."""
    cli = _load_blend_cli()
    Image.new("RGB", (10, 8), (10, 20, 30)).save(tmp_path / "a.png")
    Image.new("RGB", (10, 8), (200, 150, 100)).save(tmp_path / "b.png")
    out = tmp_path / "avg.tif"
    rc = cli.main(
        ["--mode", "average", "--out", str(out),
         str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    )
    assert rc == 0
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert printed.startswith("OK ")           # machine-readable success line
    arr = iio.imread(out)
    assert arr.shape == (8, 10, 3) and arr.dtype == np.uint16


def test_blend_cli_rejects_single_image(tmp_path: Path) -> None:
    cli = _load_blend_cli()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "solo.png")
    out = tmp_path / "x.tif"
    with __import__("pytest").raises(SystemExit):
        cli.main(["--mode", "multiply", "--out", str(out), str(tmp_path / "solo.png")])


def test_blend_folder_end_to_end(tmp_path: Path, capsys) -> None:
    # a.png dark, b.png bright — sorted by name, canon_bright must pick b.
    Image.new("RGB", (12, 10), (10, 20, 30)).save(tmp_path / "a.png")
    Image.new("RGB", (12, 10), (200, 150, 100)).save(tmp_path / "b.png")
    out = tmp_path / "result.tif"

    cli = _load_cli()
    rc = cli.main([str(tmp_path), "--mode", "canon_bright", "--out", str(out)])
    assert rc == 0
    assert out.is_file()

    back = iio.imread(out)
    assert back.dtype == np.uint16
    assert back.shape == (10, 12, 3)
    expected = np.round(np.array([200, 150, 100]) / 255 * 65535).astype(np.uint16)
    assert np.array_equal(back[0, 0], expected)
    assert "Wrote" in capsys.readouterr().out


def test_blend_folder_softness_and_dark(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), (60, 60, 60)).save(tmp_path / "a.png")
    Image.new("RGB", (8, 8), (180, 180, 180)).save(tmp_path / "b.png")
    out = tmp_path / "soft.tif"
    cli = _load_cli()
    rc = cli.main([
        str(tmp_path), "--mode", "canon_dark",
        "--softness", "40", "--bias", "5", "--basis", "luminance",
        "--out", str(out),
    ])
    assert rc == 0
    back = iio.imread(out)
    lo = round(60 / 255 * 65535)
    hi = round(180 / 255 * 65535)
    assert np.all(back >= lo) and np.all(back <= hi)  # soft blend stays bounded
