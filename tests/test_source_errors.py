"""Clear errors when a source file can't be read (e.g. a disconnected drive).

Regression for: export showed the opaque ``b'Input/output error'`` (a raw
LibRaw error) with no file name after the external drive holding the source
images was unplugged mid-session.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blendstack.core import engine
from blendstack.core import io as bs_io


def _png(path: Path, value: int = 100) -> Path:
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(path)
    return path


def test_check_sources_passes_for_readable_files(tmp_path):
    bs_io.check_sources([_png(tmp_path / "a.png"), _png(tmp_path / "b.png")])


def test_check_sources_names_missing_files_and_hints_at_drive(tmp_path):
    ok = _png(tmp_path / "ok.png")
    gone = tmp_path / "vanished" / "IMG_0916.CR2"      # as if a volume was unplugged
    with pytest.raises(FileNotFoundError) as info:
        bs_io.check_sources([ok, gone])
    msg = str(info.value)
    assert "IMG_0916.CR2" in msg                        # names the file
    assert str(gone.parent) in msg                      # and where it was
    assert "ok.png" not in msg                          # only lists the bad ones
    assert "connected" in msg                           # tells the user what to check
    assert "b'" not in msg                              # no raw bytes repr


def test_check_sources_lists_every_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        bs_io.check_sources([tmp_path / "x.tif", tmp_path / "y.cr3"])
    assert "x.tif" in str(info.value) and "y.cr3" in str(info.value)


def test_blend_files_fails_fast_before_any_work(tmp_path):
    a = _png(tmp_path / "a.png")
    b = _png(tmp_path / "b.png")
    b.unlink()                                          # drive "disconnected" after adding
    out = tmp_path / "out.tif"
    t = time.monotonic()
    with pytest.raises(FileNotFoundError, match="b.png"):
        engine.blend_files([a, b], out_path=out)
    assert time.monotonic() - t < 1.0
    assert not out.exists()                             # nothing half-written


def test_raw_open_failure_is_readable_not_bytes(tmp_path):
    pytest.importorskip("rawpy")
    junk = tmp_path / "IMG_0001.CR2"
    junk.write_bytes(b"this is not a raw file" * 100)
    for fn in (bs_io.load_image, bs_io.probe_size):
        with pytest.raises(OSError) as info:
            fn(junk)
        msg = str(info.value)
        assert "IMG_0001.CR2" in msg                    # names the file
        assert "b'" not in msg                          # decoded, not a bytes repr
        assert "connected" in msg


def test_clean_library_message_decodes_bytes():
    assert bs_io._clean_library_message(Exception(b"Input/output error")) == "Input/output error"
    assert bs_io._clean_library_message(Exception("b'Input/output error'")) == "Input/output error"
    assert bs_io._clean_library_message(Exception("plain text")) == "plain text"
