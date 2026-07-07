"""I/O tests — project brief §8 (round-trip, GIF frame 0, alpha flatten,
RAW loading) plus output-encoding checks for §4.4."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from blendstack.core import engine
from blendstack.core import io as bs_io

RNG = np.random.default_rng(42)


def _random_uint16(h: int = 64, w: int = 48) -> np.ndarray:
    return RNG.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)


# --------------------------------------------------------------------------
# §8: 16-bit TIFF round-trip through the pipeline is lossless
# --------------------------------------------------------------------------

class TestTiffRoundTrip:
    def test_pipeline_round_trip_bit_identical(self, tmp_path: Path) -> None:
        """16-bit TIFF in -> pipeline with no adjustments -> 16-bit TIFF out.

        The engine enforces a 2-image minimum per blend (brief §2), so the
        'single image through the pipeline' is realised as the image blended
        with itself at defaults — max(A, A) == A exactly, i.e. the identical
        full pipeline with a mathematically transparent fold step.
        """
        original = _random_uint16()
        src = tmp_path / "src.tif"
        iio.imwrite(src, original)  # independent writer for the input file

        dst = tmp_path / "out.tif"
        engine.blend_files([src, src], mode="canon_bright", out_path=dst)

        back = iio.imread(dst)
        assert back.dtype == np.uint16
        assert np.array_equal(back, original), "16-bit TIFF round trip must be lossless"

    def test_load_save_identity_is_lossless(self, tmp_path: Path) -> None:
        original = _random_uint16()
        src = tmp_path / "src.tif"
        iio.imwrite(src, original)

        loaded = bs_io.load_image(src)  # float32 0..1
        assert loaded.dtype == np.float32
        dst = bs_io.save_image(loaded, tmp_path / "out.tif")
        assert np.array_equal(iio.imread(dst), original)

    def test_extreme_values_survive(self, tmp_path: Path) -> None:
        original = np.zeros((4, 4, 3), dtype=np.uint16)
        original[0] = 65535
        original[1] = 1
        original[2] = 32768
        src = tmp_path / "src.tif"
        iio.imwrite(src, original)
        dst = bs_io.save_image(bs_io.load_image(src), tmp_path / "out.tif")
        assert np.array_equal(iio.imread(dst), original)


# --------------------------------------------------------------------------
# §8: GIF loads frame 0; alpha PNG flattens against black
# --------------------------------------------------------------------------

class TestLoading:
    def test_gif_loads_first_frame(self, tmp_path: Path) -> None:
        red = Image.new("RGB", (10, 8), (255, 0, 0))
        blue = Image.new("RGB", (10, 8), (0, 0, 255))
        path = tmp_path / "anim.gif"
        red.save(path, save_all=True, append_images=[blue], duration=100, loop=0)

        arr = bs_io.load_image(path)
        assert arr.shape == (8, 10, 3)
        assert np.allclose(arr, [1.0, 0.0, 0.0], atol=1 / 255), "expected frame 0 (red)"

    def test_alpha_png_flattens_against_black(self, tmp_path: Path) -> None:
        rgba = np.zeros((6, 6, 4), dtype=np.uint8)
        rgba[..., 0] = 200  # red
        rgba[..., 1] = 100  # green
        rgba[..., 2] = 50   # blue
        rgba[..., 3] = 128  # ~50% alpha
        path = tmp_path / "alpha.png"
        Image.fromarray(rgba, "RGBA").save(path)

        arr = bs_io.load_image(path)
        alpha = 128 / 255
        expected = np.array([200, 100, 50], dtype=np.float32) / 255 * alpha
        assert np.allclose(arr[0, 0], expected, atol=1e-6)

    def test_fully_transparent_becomes_black(self, tmp_path: Path) -> None:
        rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
        rgba[..., 3] = 0
        path = tmp_path / "clear.png"
        Image.fromarray(rgba, "RGBA").save(path)
        assert np.array_equal(bs_io.load_image(path), np.zeros((4, 4, 3), np.float32))

    def test_greyscale_and_palette_images(self, tmp_path: Path) -> None:
        grey = Image.new("L", (5, 4), 128)
        p_grey = tmp_path / "grey.png"
        grey.save(p_grey)
        arr = bs_io.load_image(p_grey)
        assert arr.shape == (4, 5, 3)
        assert np.allclose(arr, 128 / 255, atol=1e-6)

        pal = Image.new("RGB", (5, 4), (10, 200, 30)).convert("P")
        p_pal = tmp_path / "pal.gif"
        pal.save(p_pal)
        arr = bs_io.load_image(p_pal)
        assert arr.shape == (4, 5, 3)


_RAW_SAMPLE = os.environ.get("BLENDSTACK_RAW_SAMPLE", "")


class TestRawLoading:
    @pytest.mark.skipif(
        not (_RAW_SAMPLE and Path(_RAW_SAMPLE).is_file()),
        reason="no RAW sample available; set BLENDSTACK_RAW_SAMPLE=/path/to/file.cr3",
    )
    def test_raw_loads_via_libraw_defaults(self) -> None:
        path = Path(_RAW_SAMPLE)
        arr = bs_io.load_image(path)
        assert arr.dtype == np.float32
        assert arr.ndim == 3 and arr.shape[2] == 3
        assert arr.min() >= 0.0 and arr.max() <= 1.0
        w, h = bs_io.probe_size(path)
        assert (h, w) == arr.shape[:2]

    def test_raw_extensions_are_recognised(self) -> None:
        for ext in (".cr2", ".cr3", ".nef", ".arw", ".dng"):
            assert ext in bs_io.RAW_EXTENSIONS


# --------------------------------------------------------------------------
# §4.4: output encodings
# --------------------------------------------------------------------------

class TestSaving:
    def test_png_16bit_output(self, tmp_path: Path) -> None:
        original = _random_uint16(20, 30)
        dst = bs_io.save_image(original.astype(np.float32) / 65535.0,
                               tmp_path / "out.png")
        with Image.open(dst) as im:
            assert im.size == (30, 20)
        # Verify actual 16-bit depth straight from the PNG header.
        header = dst.read_bytes()
        ihdr = header[header.index(b"IHDR") + 4:]
        assert ihdr[8] == 16 and ihdr[9] == 2  # bit depth 16, colour type RGB

    def test_jpeg_output_8bit_quality95(self, tmp_path: Path) -> None:
        arr = np.linspace(0, 1, 24 * 24 * 3, dtype=np.float32).reshape(24, 24, 3)
        dst = bs_io.save_image(arr, tmp_path / "out.jpg")
        with Image.open(dst) as im:
            assert im.mode == "RGB"
            assert im.size == (24, 24)
            # 4:4:4 requested (subsampling=0)
            from PIL import JpegImagePlugin
            assert JpegImagePlugin.get_sampling(im) == 0

    def test_tiff_has_srgb_profile(self, tmp_path: Path) -> None:
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        dst = bs_io.save_image(arr, tmp_path / "out.tif")
        # The embedded ICC profile carries the 'acsp' header signature...
        assert b"acsp" in dst.read_bytes()
        # ...and Pillow must see it under TIFF tag 34675 (InterColorProfile).
        with Image.open(dst) as im:
            icc = im.tag_v2.get(34675)
            assert icc and b"acsp" in bytes(icc)

    def test_jpeg_and_png_tagged_srgb(self, tmp_path: Path) -> None:
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        jpg = bs_io.save_image(arr, tmp_path / "out.jpg")
        with Image.open(jpg) as im:
            assert im.info.get("icc_profile"), "JPEG must embed the sRGB profile"
        png = bs_io.save_image(arr, tmp_path / "out.png")
        with Image.open(png) as im:
            assert im.info.get("srgb") == 0 and im.info.get("gamma")  # sRGB chunk

    def test_values_clip_then_quantise_with_rounding(self, tmp_path: Path) -> None:
        arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
        dst = bs_io.save_image(arr, tmp_path / "clip.tif")
        back = iio.imread(dst)
        assert tuple(back[0, 0]) == (0, round(0.5 * 65535), 65535)

    def test_default_filename_pattern(self) -> None:
        name = bs_io.default_filename("canon_bright", "tiff")
        assert name.startswith("blend_canon_bright_")
        assert name.endswith(".tif")
        stamp = name[len("blend_canon_bright_"):-len(".tif")]
        assert len(stamp) == 15 and stamp[8] == "-"  # YYYYMMDD-HHMMSS

    def test_format_inference_rejects_unknown_suffix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="infer"):
            bs_io.save_image(np.zeros((2, 2, 3), np.float32), tmp_path / "out.xyz")
