"""v1.1 placement tests: cover_scale / place / offset_px, the coverage-aware
fold, crop maths, and blend_arrays / blend_files with move + crop."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from blendstack.core import engine, geometry, io as bs_io
from blendstack.core.adjustments import Adjustments, linear_to_srgb, srgb_to_linear


def _flat(value, h: int = 20, w: int = 20) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _rand(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

class TestCoverScalePlace:
    @pytest.mark.parametrize("shape,target", [
        ((300, 400, 3), (400, 300)),    # identical size
        ((500, 800, 3), (400, 300)),    # larger, different aspect
        ((900, 500, 3), (200, 100)),    # portrait -> landscape
        ((100, 200, 3), (200, 100)),    # upscale
        ((301, 517, 3), (111, 97)),     # odd sizes
    ])
    def test_place_of_cover_scale_equals_conform(self, shape, target) -> None:
        img = _rand(shape[0], shape[1], seed=5)
        placed, mask = geometry.place(geometry.cover_scale(img, target), target, 0, 0)
        assert mask is None
        assert placed.dtype == np.float32
        assert np.array_equal(placed, geometry.conform(img, target))

    def test_cover_scale_never_smaller_than_target(self) -> None:
        for shape, target in [((500, 800, 3), (400, 300)), ((90, 51, 3), (200, 100)),
                              ((301, 517, 3), (111, 97))]:
            out = geometry.cover_scale(_rand(shape[0], shape[1], 1), target)
            assert out.shape[1] >= target[0] and out.shape[0] >= target[1]
            assert out.dtype == np.float32

    def test_cover_scale_identical_size_same_object(self) -> None:
        img = _rand(30, 40, 1)
        assert geometry.cover_scale(img, (40, 30)) is img
        placed, mask = geometry.place(img, (40, 30))
        assert placed is img and mask is None

    def test_cover_scale_does_not_crop(self) -> None:
        out = geometry.cover_scale(_rand(100, 400, 1), (100, 100))  # wide image
        assert out.shape[:2] == (100, 400)  # scale 1.0, no crop

    def test_offset_px(self) -> None:
        assert geometry.offset_px(0.0, 0.0, (400, 300)) == (0, 0)
        assert geometry.offset_px(0.25, -0.1, (400, 300)) == (100, -30)
        assert geometry.offset_px(0.0013, 0.5, (400, 300)) == (1, 150)
        ox, oy = geometry.offset_px(0.3, 0.3, (10, 10))
        assert isinstance(ox, int) and isinstance(oy, int)

    def test_move_exact_content_and_mask(self) -> None:
        img = _rand(40, 60, 2)
        canvas, mask = geometry.place(img, (60, 40), 10, 5)  # right 10, down 5
        assert mask.shape == (40, 60, 1) and mask.dtype == np.float32
        assert canvas.shape == (40, 60, 3) and canvas.dtype == np.float32
        assert np.array_equal(canvas[5:, 10:], img[:35, :50])
        assert np.all(canvas[:5] == 0) and np.all(canvas[:, :10] == 0)
        assert np.all(mask[5:, 10:] == 1.0)
        assert np.all(mask[:5] == 0.0) and np.all(mask[:, :10] == 0.0)

    def test_move_negative_direction(self) -> None:
        img = _rand(40, 60, 2)
        canvas, mask = geometry.place(img, (60, 40), -7, -3)
        assert np.array_equal(canvas[:37, :53], img[3:, 7:])
        assert np.all(mask[37:] == 0) and np.all(mask[:, 53:] == 0)

    def test_overscan_slides_back_into_view(self) -> None:
        # 100x40 scaled layer on a 60x40 canvas: 20 px of overscan each side.
        img = _rand(40, 100, 3)
        canvas, mask = geometry.place(img, (60, 40), 15, 0)
        assert mask is None  # still fully covered thanks to overscan
        centred, _ = geometry.place(img, (60, 40), 0, 0)
        assert np.array_equal(centred, img[:, 20:80])
        assert np.array_equal(canvas, img[:, 5:65])  # content slid right
        # beyond the overscan a gap opens
        canvas, mask = geometry.place(img, (60, 40), 25, 0)
        assert mask is not None and np.all(mask[:, :5] == 0) and np.all(mask[:, 5:] == 1)

    def test_fractional_offsets_are_rounded(self) -> None:
        img = _rand(20, 20, 4)
        c1, _ = geometry.place(img, (20, 20), 2.4, 0)
        c2, _ = geometry.place(img, (20, 20), 2, 0)
        assert np.array_equal(c1, c2)

    def test_move_completely_off_canvas(self) -> None:
        img = _rand(20, 20, 4)
        canvas, mask = geometry.place(img, (20, 20), 40, 0)
        assert np.all(canvas == 0) and np.all(mask == 0)


# --------------------------------------------------------------------------
# Coverage-aware fold
# --------------------------------------------------------------------------

def _gap_mask(h=20, w=20, left=0, right=0) -> np.ndarray:
    m = np.ones((h, w, 1), np.float32)
    if left:
        m[:, :left] = 0
    if right:
        m[:, w - right :] = 0
    return m


class TestCoverageFold:
    def test_canon_dark_gap_does_not_turn_black(self) -> None:
        a, b = _flat(0.6), _flat(0.3)
        out = engine.blend_arrays(
            [a, b], [None, Adjustments(move_x=0.25)], mode="canon_dark")
        assert np.allclose(out[:, :5], 0.6)      # gap shows the base, not black
        assert np.allclose(out[:, 5:], 0.3)      # min where both cover

    def test_canon_bright_moved_layer(self) -> None:
        a, b = _flat(0.3), _flat(0.6)
        out = engine.blend_arrays(
            [a, b], [None, Adjustments(move_x=-0.25)], mode="canon_bright")
        assert np.allclose(out[:, 15:], 0.3)
        assert np.allclose(out[:, :15], 0.6)

    def test_multiply_gap_keeps_base(self) -> None:
        a, b = _flat(0.5), _flat(0.5)
        out = engine.blend_arrays(
            [a, b], [None, Adjustments(move_y=0.5)], mode="multiply")
        assert np.allclose(out[:10], 0.5)
        assert np.allclose(out[10:], 0.25, atol=1e-6)

    def test_moved_base_layer_gap_filled_by_next(self) -> None:
        a, b = _flat(0.6), _flat(0.3)
        out = engine.blend_arrays(
            [a, b], [Adjustments(move_x=0.25), None], mode="canon_dark")
        assert np.allclose(out[:, :5], 0.3)      # only b covers the gap
        assert np.allclose(out[:, 5:], 0.3)      # min(0.6, 0.3)
        out = engine.blend_arrays(
            [a, b], [Adjustments(move_x=0.25), None], mode="canon_bright")
        assert np.allclose(out[:, :5], 0.3) and np.allclose(out[:, 5:], 0.6)

    def test_uncovered_by_all_is_black(self) -> None:
        a, b = _flat(0.6), _flat(0.8)
        adj = Adjustments(move_x=0.25)
        out = engine.blend_arrays([a, b], [adj, adj], mode="canon_bright")
        assert np.all(out[:, :5] == 0.0)
        assert np.allclose(out[:, 5:], 0.8)
        assert out.dtype == np.float32

    def test_incoming_only_pixels_ignore_opacity(self) -> None:
        a, b = _flat(0.2), _flat(0.9)
        out = engine.blend_arrays(
            [a, b], [Adjustments(move_x=0.25), Adjustments(opacity=50.0)],
            mode="canon_bright")
        assert np.allclose(out[:, :5], 0.9)                  # as is
        assert np.allclose(out[:, 5:], 0.2 + (0.9 - 0.2) * 0.5, atol=1e-6)

    def test_average_three_layers_per_pixel_counts(self) -> None:
        a, b, c = _flat(0.2), _flat(0.5), _flat(0.8)
        out = engine.blend_arrays(
            [a, b, c],
            [None, Adjustments(move_x=0.25), Adjustments(move_x=-0.25)],
            mode="average")
        lin = [float(srgb_to_linear(np.float32(v))) for v in (0.2, 0.5, 0.8)]

        def enc(vals):
            return float(linear_to_srgb(np.float32(np.mean(vals))))

        # +x shift 5 px: b covers cols 5..19; c shifted left: covers 0..14.
        assert np.allclose(out[:, 0:5], enc([lin[0], lin[2]]), atol=1e-5)
        assert np.allclose(out[:, 5:15], enc(lin), atol=1e-5)
        assert np.allclose(out[:, 15:20], enc([lin[0], lin[1]]), atol=1e-5)

    def test_average_vertical_and_horizontal_shifts(self) -> None:
        imgs = [_flat(0.2), _flat(0.5), _flat(0.8)]
        advs = [None, Adjustments(move_x=0.5), Adjustments(move_y=0.5)]
        out = engine.blend_arrays(imgs, advs, mode="average")
        lin = [float(srgb_to_linear(np.float32(v))) for v in (0.2, 0.5, 0.8)]
        enc = lambda v: float(linear_to_srgb(np.float32(np.mean(v))))  # noqa: E731
        assert np.allclose(out[:10, :10], enc([lin[0]]), atol=1e-5)          # a only
        assert np.allclose(out[:10, 10:], enc([lin[0], lin[1]]), atol=1e-5)  # a, b
        assert np.allclose(out[10:, :10], enc([lin[0], lin[2]]), atol=1e-5)  # a, c
        assert np.allclose(out[10:, 10:], enc(lin), atol=1e-5)               # all

    def test_average_accepts_int_and_array_counts(self) -> None:
        from blendstack.core.modes import get_mode
        mode = get_mode("average")
        acc, inc = _flat(0.2), _flat(0.6)
        by_int = mode.blend(acc, inc, {}, count=3)
        by_arr = mode.blend(acc, inc, {}, count=np.full((20, 20, 1), 3, np.float32))
        assert np.array_equal(by_int, by_arr)
        by_zero = mode.blend(acc, inc, {}, count=np.zeros((20, 20, 1), np.float32))
        assert np.allclose(by_zero, (acc + inc) / 2)  # count clamped to >= 1

    def test_all_ones_mask_matches_fast_path_bit_exactly(self) -> None:
        imgs = [_rand(16, 24, s) for s in range(4)]
        for mode in ("canon_bright", "average", "multiply"):
            plain = engine.fold_images(imgs, mode=mode, opacities=[100, 60, 100, 30])
            masked = engine.fold_images(
                imgs, mode=mode, opacities=[100, 60, 100, 30],
                masks=[np.ones((16, 24, 1), np.float32)] * 4)
            assert np.array_equal(plain, masked)

    def test_entering_coverage_mode_midway_keeps_counts(self) -> None:
        # Two full layers, then a partial third: average must weight by 3 in
        # the covered area and leave the rest as the mean of two.
        imgs = [_flat(0.2), _flat(0.5), _flat(0.8)]
        masks = [None, None, _gap_mask(left=10)]
        out = engine.fold_images(imgs, mode="average", masks=masks)
        lin = [float(srgb_to_linear(np.float32(v))) for v in (0.2, 0.5, 0.8)]
        assert np.allclose(out[:, :10], linear_to_srgb(np.float32(np.mean(lin[:2]))),
                           atol=1e-5)
        assert np.allclose(out[:, 10:], linear_to_srgb(np.float32(np.mean(lin))),
                           atol=1e-5)

    def test_mask_shape_validated(self) -> None:
        fold = engine.BlendFold("canon_bright")
        with pytest.raises(ValueError, match="mask"):
            fold.push(_flat(0.5), mask=np.ones((5, 5, 1), np.float32))

    def test_fold_coverage_property(self) -> None:
        fold = engine.BlendFold("canon_bright")
        fold.push(_flat(0.5))
        assert fold.coverage is None
        fold.push(_flat(0.6), mask=_gap_mask(left=5))
        assert fold.coverage is None  # base fully covered, still all covered
        fold2 = engine.BlendFold("canon_bright")
        fold2.push(_flat(0.5), mask=_gap_mask(left=5))
        assert fold2.coverage is not None and not fold2.coverage[:, :5].any()

    def test_dark_never_black_with_random_shifts(self) -> None:
        imgs = [np.clip(_rand(24, 32, s) * 0.5 + 0.3, 0, 1) for s in range(4)]
        advs = [Adjustments(move_x=0.0), Adjustments(move_x=0.2, move_y=0.1),
                Adjustments(move_x=-0.3), Adjustments(move_y=-0.25)]
        out = engine.blend_arrays(imgs, advs, mode="canon_dark")
        # base layer covers the whole canvas -> nothing may come out black
        assert out.min() >= 0.3 - 1e-6


# --------------------------------------------------------------------------
# Crop
# --------------------------------------------------------------------------

class TestCrop:
    def test_box_maths(self) -> None:
        assert engine.crop_box_px((0.0, 0.0, 1.0, 1.0), 100, 50) == (0, 0, 100, 50)
        assert engine.crop_box_px((0.1, 0.2, 0.6, 0.8), 200, 100) == (20, 20, 120, 80)
        # rounding to nearest
        assert engine.crop_box_px((0.333, 0.0, 0.667, 1.0), 100, 10) == (33, 0, 67, 10)

    def test_at_least_one_pixel(self) -> None:
        x0, y0, x1, y1 = engine.crop_box_px((0.5, 0.5, 0.5001, 0.5001), 100, 100)
        assert x1 - x0 >= 1 and y1 - y0 >= 1
        assert engine.crop_box_px((0.999, 0.999, 1.0, 1.0), 100, 100) == (99, 99, 100, 100)
        assert engine.crop_box_px((0.0, 0.0, 0.001, 0.001), 100, 100) == (0, 0, 1, 1)

    @pytest.mark.parametrize("bad", [
        (0.5, 0.0, 0.5, 1.0), (0.6, 0.0, 0.4, 1.0), (0.0, 0.7, 1.0, 0.7),
        (-0.1, 0.0, 1.0, 1.0), (0.0, 0.0, 1.2, 1.0), (0.0, 0.0, 1.0),
        ("a", 0, 1, 1),
    ])
    def test_invalid_crop_raises(self, bad) -> None:
        with pytest.raises(ValueError):
            engine.crop_box_px(bad, 100, 100)
        with pytest.raises(ValueError):
            engine.apply_crop(np.zeros((10, 10, 3), np.float32), bad)

    def test_apply_crop_dims_content_and_contiguity(self) -> None:
        comp = _rand(60, 100, 6)
        out = engine.apply_crop(comp, (0.1, 0.25, 0.6, 0.75))
        x0, y0, x1, y1 = engine.crop_box_px((0.1, 0.25, 0.6, 0.75), 100, 60)
        assert out.shape == (y1 - y0, x1 - x0, 3) == (30, 50, 3)
        assert out.flags.c_contiguous and out.base is None
        assert np.array_equal(out, comp[y0:y1, x0:x1])
        out[...] = 0
        assert comp.max() > 0  # a copy, not a view

    def test_none_crop_returns_input(self) -> None:
        comp = _rand(10, 10, 1)
        assert engine.apply_crop(comp, None) is comp


# --------------------------------------------------------------------------
# blend_arrays / blend_files: bit-identical defaults, move + crop
# --------------------------------------------------------------------------

def _reference(images, adjustments, mode, params=None):
    """Pre-v1.1 pipeline: conform -> adjust -> push."""
    adjs = [Adjustments.from_mapping(a) for a in (adjustments or [None] * len(images))]
    conformed = engine.conform_images(images)
    fold = engine.BlendFold(mode, params)
    for im, a in zip(conformed, adjs):
        fold.push(engine.adjust_image(im, a), opacity=a.opacity)
    return fold.result()


class TestDefaultPathBitIdentical:
    @pytest.mark.parametrize("mode", ["canon_bright", "canon_dark", "average",
                                      "multiply", "screen", "overlay"])
    def test_matches_straight_reference_mixed_sizes(self, mode) -> None:
        images = [_rand(60, 90, 1), _rand(80, 120, 2), _rand(70, 70, 3),
                  _rand(60, 90, 4)]
        out = engine.blend_arrays(images, mode=mode)
        assert np.array_equal(out, _reference(images, None, mode))

    def test_matches_reference_with_pixelwise_adjustments_and_opacity(self) -> None:
        images = [_rand(60, 90, 1), _rand(80, 120, 2), _rand(90, 70, 3)]
        adjs = [None,
                {"exposure": 0.7, "contrast": 30.0, "saturation": 25.0,
                 "opacity": 60.0,
                 "curve_master": [[0, 0], [0.5, 0.6], [1, 1]]},
                {"opacity": 80.0, "saturation": -40.0}]
        out = engine.blend_arrays(images, adjs, mode="canon_bright",
                                  params={"softness": 10})
        ref = _reference(images, adjs, "canon_bright", {"softness": 10})
        assert np.array_equal(out, ref)

    def test_same_size_identity_path_returns_equal(self) -> None:
        images = [_rand(30, 40, 1), _rand(30, 40, 2)]
        out = engine.blend_arrays(images, mode="canon_bright")
        assert np.array_equal(out, np.maximum(images[0], images[1]))

    def test_fold_images_signature_still_works(self) -> None:
        images = [_rand(10, 12, 1), _rand(10, 12, 2)]
        assert np.array_equal(engine.fold_images(images, "canon_dark"),
                              np.minimum(images[0], images[1]))
        assert np.array_equal(
            engine.fold_images(images, "canon_dark", None, [100, 50]),
            _reference(images, [None, {"opacity": 50.0}], "canon_dark"))


class TestMoveAndCrop:
    def test_blend_arrays_crop_dims(self) -> None:
        images = [_flat(0.4, 60, 100), _flat(0.5, 60, 100)]
        crop = (0.1, 0.2, 0.7, 0.9)
        out = engine.blend_arrays(images, mode="canon_bright", crop=crop)
        x0, y0, x1, y1 = engine.crop_box_px(crop, 100, 60)
        assert out.shape == (y1 - y0, x1 - x0, 3) == (42, 60, 3)
        assert out.dtype == np.float32

    def test_blend_arrays_crop_matches_crop_of_uncropped(self) -> None:
        images = [_rand(60, 100, 1), _rand(60, 100, 2)]
        adjs = [None, Adjustments(move_x=0.1, move_y=-0.1)]
        full = engine.blend_arrays(images, adjs, mode="canon_dark")
        crop = (0.25, 0.1, 0.9, 0.6)
        assert np.array_equal(
            engine.blend_arrays(images, adjs, mode="canon_dark", crop=crop),
            engine.apply_crop(full, crop))

    def test_move_with_overscan_uses_hidden_content(self) -> None:
        # Layer b is wider than the target: moving it must reveal overscan
        # pixels, not a black gap.
        a = _flat(0.0, 40, 40)
        b = np.zeros((40, 100, 3), np.float32)
        b[:, :20] = 1.0  # white band hidden in the left overscan (cols 0..19)
        # a is the smallest (1600 px) so target = 40x40; b scaled 1.0 (h==40).
        out = engine.blend_arrays([a, b], [None, Adjustments(move_x=0.5)],
                                  mode="canon_bright")
        # shift right by 20: canvas col c shows b col c + 30 - 20 = c + 10;
        # white (cols 0..19) is visible on canvas cols 0..9.
        assert np.allclose(out[:, :10], 1.0) and np.allclose(out[:, 10:], 0.0)

    def test_move_does_not_adjust_gaps(self) -> None:
        # exposure +3 on a moved layer must not light the transparent gap.
        a, b = _flat(0.0), _flat(0.0)
        out = engine.blend_arrays(
            [a, b], [None, Adjustments(move_x=0.5, exposure=3.0)],
            mode="canon_dark")
        assert np.all(out == 0.0)  # min(0, anything) = 0, gap stays base 0
        out = engine.blend_arrays(
            [a + 0.2, b], [None, Adjustments(move_x=0.5)], mode="canon_bright")
        assert np.allclose(out[:, :10], 0.2) and np.allclose(out[:, 10:], 0.2)

    def test_blend_files_move_and_crop(self, tmp_path: Path) -> None:
        paths = []
        for i, (h, w) in enumerate([(60, 100), (90, 130), (60, 100)]):
            data = (np.random.default_rng(i).random((h, w, 3)) * 65535).astype(np.uint16)
            p = tmp_path / f"im{i}.tif"
            iio.imwrite(p, data)
            paths.append(p)
        adjs = [None, {"move_x": 0.2, "move_y": -0.15, "exposure": 0.5},
                {"move_x": -0.3}]
        crop = (0.1, 0.1, 0.8, 0.9)
        dst = tmp_path / "out.tif"
        engine.blend_files(paths, mode="canon_dark", adjustments=adjs,
                           out_path=dst, crop=crop)
        back = iio.imread(dst)
        x0, y0, x1, y1 = engine.crop_box_px(crop, 100, 60)
        assert back.shape == (y1 - y0, x1 - x0, 3) == (48, 70, 3)

        # equals the in-memory pipeline (up to 16-bit quantisation)
        images = [bs_io.load_image(p) for p in paths]
        expected = engine.blend_arrays(images, adjs, mode="canon_dark", crop=crop)
        got = back.astype(np.float32) / 65535.0
        assert np.abs(got - expected).max() <= 1.0 / 65535.0 + 1e-6

    def test_blend_files_default_matches_blend_arrays(self, tmp_path: Path) -> None:
        paths = []
        for i, (h, w) in enumerate([(50, 80), (70, 90)]):
            data = (np.random.default_rng(10 + i).random((h, w, 3)) * 65535
                    ).astype(np.uint16)
            p = tmp_path / f"d{i}.tif"
            iio.imwrite(p, data)
            paths.append(p)
        dst = tmp_path / "o.tif"
        engine.blend_files(paths, mode="average", out_path=dst)
        images = [bs_io.load_image(p) for p in paths]
        expected = engine.blend_arrays(images, mode="average")
        got = iio.imread(dst).astype(np.float32) / 65535.0
        assert np.abs(got - expected).max() <= 1.0 / 65535.0 + 1e-6

    def test_invalid_crop_rejected_before_work(self) -> None:
        images = [_flat(0.4), _flat(0.5)]
        with pytest.raises(ValueError):
            engine.blend_arrays(images, crop=(0.5, 0.0, 0.5, 1.0))

    def test_adjustments_dict_accepts_new_keys(self) -> None:
        images = [_flat(0.4), _flat(0.5)]
        out = engine.blend_arrays(
            images, [None, {"move_x": 0.25, "denoise": 20.0, "order": ["sharpen"]}],
            mode="canon_bright")
        assert out.shape == (20, 20, 3)
