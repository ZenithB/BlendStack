"""v1.1 per-image adjustment tests: curves, denoise, stage order, new
Adjustments fields, sharpen radius range and the fast gaussian blur."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from blendstack.core import adjustments as adj
from blendstack.core.adjustments import Adjustments


def _rand(h: int = 24, w: int = 32, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


# --------------------------------------------------------------------------
# Adjustments dataclass: new fields, back-compat, round trip, key
# --------------------------------------------------------------------------

OLD_KEYS = {"exposure": 1.0, "contrast": 20.0, "saturation": -10.0,
            "sharpen_radius": 2.0, "sharpen_amount": 50.0, "opacity": 70.0}


class TestAdjustmentsFields:
    def test_old_six_key_dict_loads_with_defaults(self) -> None:
        a = Adjustments.from_mapping(OLD_KEYS)
        assert a.exposure == 1.0 and a.opacity == 70.0
        assert a.denoise == 0.0
        assert a.curve_master == adj.IDENTITY_CURVE
        assert a.order == adj.DEFAULT_ORDER
        assert (a.move_x, a.move_y) == (0.0, 0.0)

    def test_defaults(self) -> None:
        a = Adjustments()
        assert a.denoise == 0.0 and a.move_x == 0.0 and a.move_y == 0.0
        for name in ("curve_master", "curve_red", "curve_green", "curve_blue"):
            assert getattr(a, name) == ((0.0, 0.0), (1.0, 1.0))
        assert adj.STAGES == ("exposure", "contrast", "curves", "saturation",
                              "denoise", "sharpen")
        assert adj.DEFAULT_ORDER == adj.STAGES
        assert a.is_identity

    def test_json_round_trip_of_asdict(self) -> None:
        a = Adjustments(
            exposure=0.5, denoise=40.0, move_x=0.1, move_y=-0.2,
            curve_master=((0.0, 0.0), (0.4, 0.5), (1.0, 1.0)),
            curve_red=((0.0, 0.1), (1.0, 0.9)),
            order=("sharpen", "denoise", "curves", "saturation", "contrast",
                   "exposure"),
        )
        text = json.dumps(dataclasses.asdict(a))
        back = Adjustments.from_mapping(json.loads(text))
        assert back == a
        assert isinstance(back.curve_master[1], tuple)
        assert isinstance(back.order, tuple)

    def test_unknown_key_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown adjustment"):
            Adjustments.from_mapping({"exposure": 1.0, "bogus": 3})

    def test_hashable(self) -> None:
        assert hash(Adjustments()) == hash(Adjustments())
        assert {Adjustments(): 1}[Adjustments()] == 1

    @pytest.mark.parametrize("bad,match", [
        ([(0.0, 0.0)], "at least 2"),
        ([(0.0, 0.0), (0.0, 1.0)], "strictly increasing"),
        ([(0.0, 0.0), (1.5, 1.0)], "0..1"),
        ([(0.0, -0.1), (1.0, 1.0)], "0..1"),
        ([(0.0, 0.0), (float("nan"), 1.0)], "0..1"),
        ([(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)], "pairs"),
    ])
    def test_bad_curves_rejected(self, bad, match) -> None:
        with pytest.raises(ValueError, match=match):
            Adjustments.from_mapping({"curve_master": bad})

    def test_unsorted_curve_points_are_sorted(self) -> None:
        a = Adjustments.from_mapping(
            {"curve_red": [[1.0, 1.0], [0.5, 0.2], [0.0, 0.0]]})
        assert a.curve_red == ((0.0, 0.0), (0.5, 0.2), (1.0, 1.0))

    def test_order_normalisation(self) -> None:
        a = Adjustments.from_mapping({"order": ["sharpen", "bogus", "exposure",
                                                "sharpen"]})
        assert a.order == ("sharpen", "exposure", "contrast", "curves",
                           "saturation", "denoise")
        assert set(a.order) == set(adj.STAGES) and len(a.order) == 6
        assert Adjustments.from_mapping({"order": []}).order == adj.DEFAULT_ORDER

    def test_is_identity_considers_new_fields(self) -> None:
        assert not Adjustments(denoise=1.0).is_identity
        assert not Adjustments(curve_blue=((0.0, 0.0), (0.5, 0.6), (1.0, 1.0))
                               ).is_identity
        # opacity / placement / order never matter
        assert Adjustments(opacity=10.0, move_x=0.3, order=("sharpen",)
                           ).is_identity
        # a 3-point diagonal is still the identity
        assert Adjustments(curve_master=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))
                           ).is_identity

    def test_adjust_key(self) -> None:
        base = Adjustments(exposure=1.0)
        assert hash(base.adjust_key()) == hash(Adjustments(exposure=1.0).adjust_key())
        # excluded
        assert base.adjust_key() == dataclasses.replace(
            base, opacity=5.0, move_x=0.5, move_y=-0.5).adjust_key()
        # included
        for change in ({"exposure": 2.0}, {"contrast": 1.0}, {"saturation": 1.0},
                       {"denoise": 1.0}, {"sharpen_radius": 3.0},
                       {"sharpen_amount": 1.0},
                       {"curve_master": ((0.0, 0.1), (1.0, 1.0))},
                       {"curve_red": ((0.0, 0.1), (1.0, 1.0))},
                       {"curve_green": ((0.0, 0.1), (1.0, 1.0))},
                       {"curve_blue": ((0.0, 0.1), (1.0, 1.0))},
                       {"order": ("contrast", "exposure")}):
            assert dataclasses.replace(base, **change).adjust_key() != base.adjust_key()


# --------------------------------------------------------------------------
# Curve LUT
# --------------------------------------------------------------------------

class TestCurveLut:
    def test_identity_lut_is_diagonal(self) -> None:
        lut = adj.curve_lut(adj.IDENTITY_CURVE)
        assert lut.dtype == np.float32 and lut.shape == (4096,)
        assert np.allclose(lut, np.linspace(0, 1, 4096), atol=1e-6)

    def test_size_parameter(self) -> None:
        assert adj.curve_lut(adj.IDENTITY_CURVE, size=256).shape == (256,)

    def test_endpoints_and_interpolation_through_points(self) -> None:
        pts = ((0.0, 0.05), (0.3, 0.5), (0.7, 0.6), (1.0, 0.95))
        lut = adj.curve_lut(pts, size=4097)  # grid hits 0.3/0.7 exactly-ish
        assert lut[0] == pytest.approx(0.05) and lut[-1] == pytest.approx(0.95)
        for x, y in pts:
            assert float(np.interp(x, np.linspace(0, 1, 4097), lut)) == pytest.approx(
                y, abs=2e-4)

    def test_monotone_for_monotone_points(self) -> None:
        pts = ((0.0, 0.0), (0.1, 0.4), (0.15, 0.42), (0.6, 0.45), (0.9, 0.99),
               (1.0, 1.0))
        assert np.all(np.diff(adj.curve_lut(pts)) >= -1e-7)

    def test_no_overshoot(self) -> None:
        # Steep step-like and non-monotone control sets.
        for pts in (((0.0, 0.0), (0.49, 0.02), (0.51, 0.98), (1.0, 1.0)),
                    ((0.0, 0.2), (0.3, 0.9), (0.6, 0.1), (1.0, 0.8)),
                    ((0.0, 0.0), (0.5, 0.0), (1.0, 1.0))):
            lut = adj.curve_lut(pts)
            ys = [p[1] for p in pts]
            assert lut.min() >= min(ys) - 1e-6 and lut.max() <= max(ys) + 1e-6
            # per segment: within the segment's end ys (monotone cubic property
            # for monotone segments; locally bounded for extrema)
            xs = np.linspace(0, 1, lut.size)
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                seg = lut[(xs >= x0) & (xs <= x1)]
                assert seg.min() >= min(y0, y1) - 1e-5
                assert seg.max() <= max(y0, y1) + 1e-5

    def test_flat_outside_end_points(self) -> None:
        lut = adj.curve_lut(((0.25, 0.3), (0.75, 0.8)))
        xs = np.linspace(0, 1, lut.size)
        assert np.all(lut[xs <= 0.25] == np.float32(0.3))
        assert np.all(lut[xs >= 0.75] == np.float32(0.8))

    def test_smooth_not_piecewise_linear(self) -> None:
        lut = adj.curve_lut(((0.0, 0.0), (0.5, 0.8), (1.0, 1.0)))
        second = np.diff(lut, 2)
        assert np.abs(second).max() < 1e-3  # no kinks at 4096 samples


class TestApplyCurves:
    def test_all_identity_returns_same_object(self) -> None:
        img = _rand()
        ident = adj.IDENTITY_CURVE
        assert adj.apply_curves(img, ident, ident, ident, ident) is img

    def test_identity_lut_values_preserved(self) -> None:
        img = _rand()
        ident = adj.IDENTITY_CURVE
        out = adj.apply_curves(img, ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
                               ident, ident, ident)
        assert out is img  # 3-point diagonal is identity too

    def test_master_applies_to_all_channels(self) -> None:
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = adj.apply_curves(img, ((0.0, 0.0), (0.5, 0.8), (1.0, 1.0)),
                               *([adj.IDENTITY_CURVE] * 3))
        assert np.allclose(out, 0.8, atol=1e-3)

    def test_per_channel_independence(self) -> None:
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = adj.apply_curves(img, adj.IDENTITY_CURVE,
                               ((0.0, 0.0), (0.5, 0.9), (1.0, 1.0)),
                               adj.IDENTITY_CURVE, adj.IDENTITY_CURVE)
        assert np.allclose(out[..., 0], 0.9, atol=1e-3)
        assert np.array_equal(out[..., 1], img[..., 1])
        assert np.array_equal(out[..., 2], img[..., 2])

    def test_master_then_channel_order(self) -> None:
        img = np.full((2, 2, 3), 0.5, dtype=np.float32)
        master = ((0.0, 0.0), (0.5, 0.8), (1.0, 1.0))   # 0.5 -> 0.8
        red = ((0.0, 0.0), (0.5, 0.1), (0.8, 0.4), (1.0, 1.0))  # 0.8 -> 0.4
        out = adj.apply_curves(img, master, red, adj.IDENTITY_CURVE,
                               adj.IDENTITY_CURVE)
        assert out[0, 0, 0] == pytest.approx(0.4, abs=2e-3)  # master first
        # if channel ran first it would be 0.1 -> master(0.1) ~ 0.16, not 0.4
        assert out[0, 0, 1] == pytest.approx(0.8, abs=2e-3)

    def test_input_clipped_and_output_in_range(self) -> None:
        img = np.array([[[-0.5, 0.5, 1.7]]], dtype=np.float32)
        out = adj.apply_curves(img, ((0.0, 0.1), (1.0, 0.9)),
                               *([adj.IDENTITY_CURVE] * 3))
        assert np.allclose(out[0, 0], [0.1, 0.5, 0.9], atol=1e-3)
        assert out.dtype == np.float32

    def test_matches_dense_lut_reference(self) -> None:
        img = _rand(30, 40, seed=9)
        c = ((0.0, 0.0), (0.3, 0.4), (0.8, 0.7), (1.0, 1.0))
        out = adj.apply_curves(img, c, adj.IDENTITY_CURVE, adj.IDENTITY_CURVE,
                               adj.IDENTITY_CURVE)
        lut = adj.curve_lut(c, 4096)
        ref = np.interp(img, np.linspace(0, 1, 4096), lut)
        assert np.abs(out - ref).max() < 1e-5


# --------------------------------------------------------------------------
# Denoise (guided filter) — calibration target
# --------------------------------------------------------------------------

def _noisy_step(seed: int = 1, h: int = 64, w: int = 128, sigma: float = 0.03
                ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.2, np.float32)
    img[:, w // 2:] = 0.8
    return (img + rng.normal(0, sigma, img.shape)).astype(np.float32)


def _edge_width(out: np.ndarray) -> float:
    """10-90 % transition width (px) of the row-averaged step profile."""
    prof = out.mean(axis=(0, 2))
    c = out.shape[1] // 2
    seg, xs = prof[c - 12 : c + 12], np.arange(c - 12, c + 12)

    def cross(v: float) -> float:
        i = int(np.argmax(seg >= v))
        return xs[i - 1] + (v - seg[i - 1]) / (seg[i] - seg[i - 1])

    return float(cross(0.2 + 0.54) - cross(0.2 + 0.06))


FLAT = (slice(None), slice(8, 48))  # flat region well clear of the edge


class TestDenoise:
    def test_zero_strength_is_identity_object(self) -> None:
        img = _noisy_step()
        assert adj.apply_denoise(img, 0.0) is img

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_calibration_strength_50(self, seed: int) -> None:
        img = _noisy_step(seed)
        out = adj.apply_denoise(img, 50.0)
        reduction = 1.0 - out[FLAT].std() / img[FLAT].std()
        assert reduction >= 0.5, f"noise std only dropped {reduction:.0%}"
        assert _edge_width(out) <= 2.0

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_strength_100_stronger_than_50(self, seed: int) -> None:
        img = _noisy_step(seed)
        std50 = adj.apply_denoise(img, 50.0)[FLAT].std()
        out100 = adj.apply_denoise(img, 100.0)
        assert out100[FLAT].std() < std50
        assert _edge_width(out100) <= 2.0

    def test_monotone_in_strength(self) -> None:
        img = _noisy_step()
        stds = [adj.apply_denoise(img, s)[FLAT].std() for s in (10, 25, 50, 75, 100)]
        assert all(a > b for a, b in zip(stds, stds[1:]))

    def test_output_dtype_shape(self) -> None:
        img = _noisy_step()
        out = adj.apply_denoise(img, 60.0)
        assert out.dtype == np.float32 and out.shape == img.shape

    def test_constant_image_unchanged(self) -> None:
        img = np.full((16, 16, 3), 0.37, np.float32)
        assert np.allclose(adj.apply_denoise(img, 100.0), 0.37, atol=1e-6)

    def test_one_constant_channel(self) -> None:
        img = _noisy_step()
        img[..., 1] = 0.5
        out = adj.apply_denoise(img, 70.0)
        assert np.allclose(out[..., 1], 0.5, atol=1e-6)
        assert out[..., 0][FLAT].std() < img[..., 0][FLAT].std()
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("shape", [(3, 3, 3), (1, 1, 3), (1, 7, 3), (2, 5, 3)])
    def test_tiny_images(self, shape) -> None:
        img = np.random.default_rng(0).random(shape, dtype=np.float32)
        for strength in (1.0, 50.0, 100.0):
            out = adj.apply_denoise(img, strength)
            assert out.shape == img.shape and np.isfinite(out).all()
            assert out.min() >= img.min() - 1e-5 and out.max() <= img.max() + 1e-5

    def test_banded_processing_matches_whole_image(self) -> None:
        img = _noisy_step(seed=4, h=70, w=40)
        whole = adj.apply_denoise(img, 80.0, _band_rows=1000)
        for rows in (7, 16, 33):
            assert np.allclose(adj.apply_denoise(img, 80.0, _band_rows=rows),
                               whole, atol=1e-6)

    def test_strong_edges_preserved(self) -> None:
        img = np.zeros((20, 20, 3), np.float32)
        img[:, 10:] = 1.0  # noise-free hard edge
        out = adj.apply_denoise(img, 100.0)
        assert out[:, :6].max() < 0.02 and out[:, 14:].min() > 0.98


# --------------------------------------------------------------------------
# Stage order
# --------------------------------------------------------------------------

class TestStageOrder:
    def test_default_order_matches_legacy_chain_bit_exactly(self) -> None:
        img = _rand()
        a = Adjustments(exposure=1.0, contrast=40.0, saturation=-30.0,
                        sharpen_radius=1.5, sharpen_amount=80.0)
        legacy = adj.apply_exposure(img, a.exposure)
        legacy = adj.apply_contrast(legacy, a.contrast)
        legacy = adj.apply_saturation(legacy, a.saturation)
        legacy = adj.apply_sharpen(legacy, a.sharpen_radius, a.sharpen_amount)
        legacy = np.clip(legacy, 0.0, 1.0).astype(np.float32)
        assert np.array_equal(adj.apply_adjustments(img, a), legacy)

    def test_exposure_contrast_do_not_commute(self) -> None:
        img = np.linspace(0.05, 0.95, 64, dtype=np.float32).reshape(8, 8, 1)
        img = np.repeat(img, 3, axis=2)
        a = Adjustments(exposure=1.0, contrast=60.0)
        default = adj.apply_adjustments(img, a)
        swapped = adj.apply_adjustments(
            img, dataclasses.replace(a, order=("contrast", "exposure")))
        assert not np.allclose(default, swapped, atol=1e-3)
        manual = np.clip(adj.apply_exposure(adj.apply_contrast(img, 60.0), 1.0),
                         0, 1)
        assert np.allclose(swapped, manual, atol=1e-6)

    def test_curves_position_matters(self) -> None:
        img = _rand()
        curve = ((0.0, 0.0), (0.5, 0.8), (1.0, 1.0))
        a = Adjustments(contrast=50.0, curve_master=curve)
        first = adj.apply_adjustments(
            img, dataclasses.replace(a, order=("curves", "contrast")))
        last = adj.apply_adjustments(
            img, dataclasses.replace(a, order=("contrast", "curves")))
        assert not np.allclose(first, last, atol=1e-3)

    def test_denoise_before_vs_after_sharpen(self) -> None:
        img = _noisy_step()
        a = Adjustments(denoise=60.0, sharpen_amount=100.0)
        dn_first = adj.apply_adjustments(img, a)
        sh_first = adj.apply_adjustments(
            img, dataclasses.replace(a, order=("sharpen", "denoise")))
        assert not np.allclose(dn_first, sh_first, atol=1e-3)
        # denoise-then-sharpen leaves less flat-region noise than the reverse
        assert dn_first[FLAT].std() < sh_first[FLAT].std()

    def test_partial_order_is_completed(self) -> None:
        img = _rand()
        a = Adjustments(exposure=1.0, contrast=40.0, order=("contrast",))
        # contrast first, then the rest in default order (exposure next)
        assert a.order[0] == "contrast" and a.order[1] == "exposure"
        out = adj.apply_adjustments(img, a)
        manual = np.clip(adj.apply_exposure(adj.apply_contrast(img, 40.0), 1.0),
                         0, 1)
        assert np.allclose(out, manual, atol=1e-6)

    def test_identity_short_circuit_and_clip(self) -> None:
        img = _rand()
        assert adj.apply_adjustments(img, Adjustments(order=("sharpen",))) is img
        assert adj.apply_adjustments(img, {"curve_red": [[0, 0], [1, 1]]}) is img
        out = adj.apply_adjustments(img, Adjustments(exposure=3.0, contrast=100.0))
        assert out.min() >= 0.0 and out.max() <= 1.0 and out.dtype == np.float32

    def test_all_stages_run(self) -> None:
        img = _noisy_step()
        a = Adjustments(exposure=0.3, contrast=10.0, saturation=20.0,
                        denoise=30.0, sharpen_amount=50.0,
                        curve_green=((0.0, 0.0), (0.5, 0.6), (1.0, 1.0)))
        out = adj.apply_adjustments(img, a)
        assert out.shape == img.shape and np.isfinite(out).all()


# --------------------------------------------------------------------------
# Sharpen radius range and fast blur
# --------------------------------------------------------------------------

class TestSharpenRadiusAndBlur:
    def test_radius_20_accepted_and_above_clamped(self) -> None:
        img = _rand(60, 70, seed=3)
        r20 = adj.apply_sharpen(img, 20.0, 100.0)
        assert np.array_equal(adj.apply_sharpen(img, 25.0, 100.0), r20)
        assert np.array_equal(adj.apply_sharpen(img, 1000.0, 100.0), r20)
        assert not np.allclose(r20, adj.apply_sharpen(img, 10.0, 100.0), atol=1e-4)
        # lower clamp unchanged
        assert np.array_equal(adj.apply_sharpen(img, 0.1, 100.0),
                              adj.apply_sharpen(img, 0.5, 100.0))

    def test_small_sigma_uses_exact_kernel(self) -> None:
        img = _rand(40, 40, seed=2)
        k = adj._gaussian_kernel(2.0)
        exact = adj._convolve_axis(adj._convolve_axis(img, k, 0), k, 1)
        assert np.array_equal(adj.gaussian_blur(img, 2.0), exact)

    @pytest.mark.parametrize("sigma", [3.5, 6.0, 10.0, 20.0])
    def test_fast_blur_close_to_exact(self, sigma: float) -> None:
        rng = np.random.default_rng(3)
        h, w = 160, 200
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.stack([0.5 + 0.3 * np.sin(xx / 37) + 0.2 * np.cos(yy / 23),
                        xx / w, yy / h], axis=-1)
        img = (img + rng.normal(0, 0.05, img.shape)).clip(0, 1).astype(np.float32)
        k = adj._gaussian_kernel(sigma)
        exact = adj._convolve_axis(adj._convolve_axis(img, k, 0), k, 1)
        fast = adj.gaussian_blur(img, sigma)
        assert fast.dtype == np.float32 and fast.shape == img.shape
        assert float(np.abs(fast - exact).mean()) < 0.005

    def test_fast_blur_preserves_flat_and_energy(self) -> None:
        flat = np.full((50, 60, 3), 0.42, np.float32)
        assert np.allclose(adj.gaussian_blur(flat, 12.0), 0.42, atol=1e-5)
        imp = np.zeros((101, 101, 3), np.float32)
        imp[50, 50] = 1.0
        out = adj.gaussian_blur(imp, 8.0)
        assert float(out[..., 0].sum()) == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(out[50, :, 0], out[:, 50, 0], atol=1e-7)  # symmetric
        assert np.allclose(out[50, :, 0], out[50, ::-1, 0], atol=1e-7)

    def test_box_radii_variance_matches_gaussian(self) -> None:
        for sigma in (3.5, 7.0, 15.0, 20.0):
            radii = adj._box_radii(sigma)
            var = sum(((2 * r + 1) ** 2 - 1) / 12.0 for r in radii)
            assert var == pytest.approx(sigma * sigma, rel=0.05)
