"""Adjustment-chain tests — project brief §4.1 definitions."""

from __future__ import annotations

import numpy as np
import pytest

from blendstack.core import adjustments as adj


def _flat(value: float, shape: tuple[int, int] = (8, 8)) -> np.ndarray:
    return np.full(shape + (3,), value, dtype=np.float32)


class TestSrgbEotf:
    def test_piecewise_not_plain_gamma(self) -> None:
        # Below the toe the EOTF is linear /12.92 — a plain gamma 2.2 is not.
        x = np.float32(0.003)
        assert np.isclose(adj.srgb_to_linear(np.array(x)), x / 12.92, rtol=1e-5)
        # Above the toe it is the offset 2.4 power law.
        y = np.float32(0.5)
        assert np.isclose(adj.srgb_to_linear(np.array(y)),
                          ((y + 0.055) / 1.055) ** 2.4, rtol=1e-5)

    def test_round_trip(self) -> None:
        x = np.linspace(0, 1, 512, dtype=np.float32)
        assert np.allclose(adj.linear_to_srgb(adj.srgb_to_linear(x)), x, atol=1e-6)


class TestExposure:
    def test_plus_one_ev_doubles_linear_light(self) -> None:
        img = _flat(0.5)
        out = adj.apply_exposure(img, 1.0)
        expected = adj.linear_to_srgb(adj.srgb_to_linear(img) * 2.0)
        assert np.allclose(out, expected, atol=1e-6)
        # A gamma-space gain would give 1.0; linear-light gives ~0.6858
        # (sRGB 0.5 -> linear 0.2140, x2 -> 0.4281 -> re-encode 0.6858).
        assert 0.67 < float(out[0, 0, 0]) < 0.70

    def test_identity_returns_same_object(self) -> None:
        img = _flat(0.3)
        assert adj.apply_exposure(img, 0.0) is img


class TestContrast:
    @pytest.mark.parametrize("ui,k", [(-100.0, 0.5), (0.0, 1.0), (100.0, 2.0)])
    def test_k_mapping_endpoints(self, ui: float, k: float) -> None:
        img = _flat(0.75)
        out = adj.apply_contrast(img, ui)
        assert np.allclose(out, (0.75 - 0.5) * k + 0.5, atol=1e-6)

    def test_pivot_is_invariant(self) -> None:
        img = _flat(0.5)
        assert np.allclose(adj.apply_contrast(img, 80.0), 0.5, atol=1e-7)


class TestSaturation:
    def test_minus_100_is_rec709_greyscale(self) -> None:
        img = np.zeros((2, 2, 3), dtype=np.float32)
        img[..., 0] = 1.0  # pure red
        out = adj.apply_saturation(img, -100.0)
        assert np.allclose(out, 0.2126, atol=1e-6)
        assert np.allclose(out[..., 0], out[..., 1])

    def test_plus_100_doubles_chroma(self) -> None:
        img = np.zeros((1, 1, 3), dtype=np.float32)
        img[..., :] = (0.6, 0.4, 0.4)
        luma = 0.2126 * 0.6 + 0.7152 * 0.4 + 0.0722 * 0.4
        out = adj.apply_saturation(img, 100.0)
        assert np.allclose(out[0, 0], luma + (np.array([0.6, 0.4, 0.4]) - luma) * 2,
                           atol=1e-6)


class TestSharpen:
    def test_unsharp_mask_formula(self) -> None:
        rng = np.random.default_rng(5)
        img = rng.random((20, 20, 3), dtype=np.float32)
        out = adj.apply_sharpen(img, radius=2.0, amount=150.0)
        expected = img + 1.5 * (img - adj.gaussian_blur(img, 2.0))
        assert np.allclose(out, expected, atol=1e-6)

    def test_gaussian_blur_preserves_flat_fields(self) -> None:
        img = _flat(0.42, (16, 16))
        assert np.allclose(adj.gaussian_blur(img, 3.0), 0.42, atol=1e-6)

    def test_gaussian_kernel_normalised_and_separable(self) -> None:
        img = np.zeros((31, 31, 3), dtype=np.float32)
        img[15, 15] = 1.0
        blurred = adj.gaussian_blur(img, 2.0)
        assert np.isclose(float(blurred.sum()) / 3.0, 1.0, atol=1e-4)
        # Rotational symmetry of the separable kernel.
        assert np.allclose(blurred[15, :, 0], blurred[:, 15, 0], atol=1e-7)


class TestChain:
    def test_identity_returns_input_uncopied(self) -> None:
        img = _flat(0.5)
        assert adj.apply_adjustments(img, None) is img
        assert adj.apply_adjustments(img, {"opacity": 30.0}) is img  # fold-step only

    def test_fixed_order_exposure_before_contrast(self) -> None:
        img = _flat(0.25)
        combined = adj.apply_adjustments(img, {"exposure": 1.0, "contrast": 50.0})
        manual = np.clip(
            adj.apply_contrast(adj.apply_exposure(img, 1.0), 50.0), 0.0, 1.0
        )
        assert np.allclose(combined, manual, atol=1e-7)

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown adjustment"):
            adj.apply_adjustments(_flat(0.1), {"exposur": 1.0})

    def test_output_clipped(self) -> None:
        out = adj.apply_adjustments(_flat(0.9), {"exposure": 3.0, "contrast": 100.0})
        assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0
