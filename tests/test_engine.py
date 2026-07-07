"""Engine correctness tests — project brief §8, Phase 1 acceptance criteria.

Covers: exact max/min equivalence, per-channel colour mixing vs luminance
basis, order-(in)dependence, softness boundedness, monotone invariance,
the 2–20 image cap, opacity-at-fold semantics, the linear-light
scaffolding, and the 24 MP × 20-image blend (marked ``slow``).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from blendstack.core import engine
from blendstack.core.adjustments import linear_to_srgb, srgb_to_linear
from blendstack.core.modes import get_mode
from blendstack.core.modes.registry import BlendMode, ModeParameter, register_mode


def _gradients(h: int = 64, w: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Two synthetic gradients: horizontal ramp vs vertical ramp."""
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(1.0, 0.0, h, dtype=np.float32)
    a = np.repeat(np.tile(x, (h, 1))[..., None], 3, axis=-1)
    b = np.repeat(np.tile(y[:, None], (1, w))[..., None], 3, axis=-1)
    return a.astype(np.float32), b.astype(np.float32)


def _random_images(n: int, h: int = 32, w: int = 48, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((h, w, 3), dtype=np.float32) for _ in range(n)]


# --------------------------------------------------------------------------
# §8: Bright at defaults == np.maximum exactly; Dark == np.minimum
# --------------------------------------------------------------------------

class TestCanonDefaultsExact:
    def test_bright_equals_maximum_exactly(self) -> None:
        a, b = _gradients()
        out = engine.blend_arrays([a, b], mode="canon_bright")
        assert out.dtype == np.float32
        assert np.array_equal(out, np.maximum(a, b))

    def test_dark_equals_minimum_exactly(self) -> None:
        a, b = _gradients()
        out = engine.blend_arrays([a, b], mode="canon_dark")
        assert np.array_equal(out, np.minimum(a, b))

    def test_defaults_exact_on_random_data_multiway(self) -> None:
        images = _random_images(6)
        expected = images[0]
        for im in images[1:]:
            expected = np.maximum(expected, im)
        out = engine.blend_arrays(images, mode="canon_bright")
        assert np.array_equal(out, expected)


# --------------------------------------------------------------------------
# §8: per-channel mixing vs luminance basis on a red-vs-green pair
# --------------------------------------------------------------------------

class TestComparisonBasis:
    A = np.array([0.8, 0.2, 0.05], dtype=np.float32)  # reddish
    B = np.array([0.1, 0.9, 0.00], dtype=np.float32)  # greenish

    def _pair(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (16, 16, 3)
        return (np.broadcast_to(self.A, shape).astype(np.float32).copy(),
                np.broadcast_to(self.B, shape).astype(np.float32).copy())

    def test_per_channel_mixes_colours(self) -> None:
        a, b = self._pair()
        out = engine.blend_arrays([a, b], mode="canon_bright",
                                  params={"basis": "per_channel"})
        # r from A, g from B, b(channel) from A — a colour in neither source.
        mixed = np.array([0.8, 0.9, 0.05], dtype=np.float32)
        assert np.allclose(out[0, 0], mixed, atol=0)
        assert not np.allclose(mixed, self.A) and not np.allclose(mixed, self.B)

    def test_luminance_never_invents_colours(self) -> None:
        a, b = self._pair()
        out = engine.blend_arrays([a, b], mode="canon_bright",
                                  params={"basis": "luminance"})
        flat = out.reshape(-1, 3)
        is_a = np.all(flat == self.A, axis=1)
        is_b = np.all(flat == self.B, axis=1)
        assert np.all(is_a | is_b), "luminance basis produced a colour absent from both sources"
        # B has the higher Rec.709 luma, so B must win outright here.
        assert np.all(is_b)


# --------------------------------------------------------------------------
# §8: order-independence at defaults; order-dependence with opacity
# --------------------------------------------------------------------------

class TestOrdering:
    def test_shuffled_defaults_bit_identical(self) -> None:
        images = _random_images(5, seed=11)
        reference = engine.blend_arrays(images, mode="canon_bright")
        for perm_seed in range(3):
            order = np.random.default_rng(perm_seed).permutation(5)
            shuffled = [images[i] for i in order]
            out = engine.blend_arrays(shuffled, mode="canon_bright")
            assert np.array_equal(out, reference)

    def test_shuffle_with_partial_opacity_differs(self) -> None:
        images = _random_images(5, seed=12)
        adjustments: list[dict[str, float]] = [{} for _ in images]
        adjustments[2] = {"opacity": 50.0}  # travels with images[2]

        reference = engine.blend_arrays(images, adjustments, mode="canon_bright")
        order = [4, 2, 0, 3, 1]
        out = engine.blend_arrays([images[i] for i in order],
                                  [adjustments[i] for i in order],
                                  mode="canon_bright")
        assert out.shape == reference.shape
        assert not np.array_equal(out, reference), (
            "50% opacity must make the fold order-dependent (documented trade-off)"
        )


# --------------------------------------------------------------------------
# §8: softness > 0 stays bounded by min/max of the sources
# --------------------------------------------------------------------------

class TestSoftnessBounded:
    @pytest.mark.parametrize("mode", ["canon_bright", "canon_dark"])
    @pytest.mark.parametrize("params", [
        {"softness": 1.0},
        {"softness": 35.0},
        {"softness": 100.0},
        {"softness": 50.0, "bias": 60.0},
        {"softness": 50.0, "bias": -60.0, "basis": "luminance"},
    ])
    def test_convex_combination_bounds(self, mode: str, params: dict) -> None:
        rng = np.random.default_rng(3)
        a = rng.random((40, 40, 3), dtype=np.float32)
        b = rng.random((40, 40, 3), dtype=np.float32)
        out = engine.blend_arrays([a, b], mode=mode, params=params)
        eps = 1e-6
        assert np.all(out >= np.minimum(a, b) - eps)
        assert np.all(out <= np.maximum(a, b) + eps)


# --------------------------------------------------------------------------
# §8: monotone-invariance (hard mode, per-channel)
# --------------------------------------------------------------------------

class TestMonotoneInvariance:
    @pytest.mark.parametrize("mode", ["canon_bright", "canon_dark"])
    def test_gamma_before_equals_gamma_after(self, mode: str) -> None:
        rng = np.random.default_rng(21)
        images = [rng.random((32, 32, 3), dtype=np.float32) for _ in range(3)]
        gamma = np.float32(2.2)

        blend_then_gamma = engine.blend_arrays(images, mode=mode) ** gamma
        gamma_then_blend = engine.blend_arrays([im ** gamma for im in images], mode=mode)
        # max/min select the same winners under any monotone tone curve,
        # so the two computations pick literally the same source values.
        assert np.allclose(gamma_then_blend, blend_then_gamma, atol=1e-7)
        mismatches = np.count_nonzero(gamma_then_blend != blend_then_gamma)
        assert mismatches == 0, f"{mismatches} pixels selected different winners"


# --------------------------------------------------------------------------
# §2 / §8: 2–20 image cap
# --------------------------------------------------------------------------

class TestImageCap:
    def test_rejects_one_image(self) -> None:
        with pytest.raises(ValueError, match="2–20|2-20|images"):
            engine.blend_arrays(_random_images(1))

    def test_rejects_twenty_one_images(self) -> None:
        base = _random_images(1)[0]
        with pytest.raises(ValueError):
            engine.blend_arrays([base] * 21)

    def test_accepts_twenty_images(self) -> None:
        base = _random_images(2, h=8, w=8)
        out = engine.blend_arrays([base[i % 2] for i in range(20)])
        assert out.shape == (8, 8, 3)

    def test_fold_push_enforces_cap(self) -> None:
        fold = engine.BlendFold("canon_bright")
        image = _random_images(1, h=4, w=4)[0]
        for _ in range(engine.MAX_IMAGES):
            fold.push(image)
        with pytest.raises(ValueError):
            fold.push(image)


# --------------------------------------------------------------------------
# §4: opacity acts at the fold step; first image's opacity ignored
# --------------------------------------------------------------------------

class TestOpacityFold:
    def test_zero_opacity_incoming_is_a_no_op(self) -> None:
        a, b = _gradients(16, 16)
        out = engine.blend_arrays([a, b], [{}, {"opacity": 0.0}])
        assert np.array_equal(out, a)

    def test_half_opacity_is_lerp_of_acc_and_blend(self) -> None:
        a, b = _gradients(16, 16)
        out = engine.blend_arrays([a, b], [{}, {"opacity": 50.0}])
        expected = a + (np.maximum(a, b) - a) * np.float32(0.5)
        assert np.allclose(out, expected, atol=1e-7)

    def test_first_image_opacity_ignored(self) -> None:
        a, b = _gradients(16, 16)
        out = engine.blend_arrays([a, b], [{"opacity": 0.0}, {}])
        assert np.array_equal(out, np.maximum(a, b))


# --------------------------------------------------------------------------
# §3 design rule 2: linear-light scaffolding for needs_linear modes
# --------------------------------------------------------------------------

class TestLinearScaffolding:
    def test_needs_linear_mode_runs_in_linear_light(self) -> None:
        captured: dict[str, np.ndarray] = {}

        @register_mode
        class _LinearProbe(BlendMode):  # noqa: N801 - test-local
            name = "_test_linear_probe"
            label = "Linear probe"
            needs_linear = True
            parameters: tuple[ModeParameter, ...] = ()

            def blend(self, accumulator, incoming, params=None):  # type: ignore[override]
                captured["acc"] = accumulator
                captured["inc"] = incoming
                return incoming

        a = np.full((4, 4, 3), 0.5, dtype=np.float32)
        b = np.full((4, 4, 3), 0.25, dtype=np.float32)
        out = engine.blend_arrays([a, b], mode="_test_linear_probe")

        # The fold must have received sRGB-linearised data...
        assert np.allclose(captured["acc"], srgb_to_linear(a), atol=1e-7)
        assert np.allclose(captured["inc"], srgb_to_linear(b), atol=1e-7)
        # ...and the result must be re-encoded back to sRGB afterwards.
        assert np.allclose(out, linear_to_srgb(srgb_to_linear(b)), atol=1e-6)
        assert np.allclose(out, b, atol=1e-6)

    def test_v1_modes_declare_gamma_space(self) -> None:
        assert get_mode("canon_bright").needs_linear is False
        assert get_mode("canon_dark").needs_linear is False


# --------------------------------------------------------------------------
# §8: 20-image blend at 24 MP completes within the memory budget
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestFullScale:
    def test_twenty_images_at_24mp(self) -> None:
        import resource

        h, w = 4000, 6000  # 24 MP
        rng = np.random.default_rng(0)
        # 5 distinct 288 MB buffers cycled to 20 entries: exercises a full
        # 20-step fold while keeping the *input* footprint realistic for the
        # engine (whose streaming path never holds 20 full images anyway).
        uniques = [rng.random((h, w, 3), dtype=np.float32) for _ in range(5)]
        images = [uniques[i % 5] for i in range(20)]

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = time.perf_counter()
        out = engine.blend_arrays(images, mode="canon_bright")
        elapsed = time.perf_counter() - start
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        assert out.shape == (h, w, 3)
        assert out.dtype == np.float32
        expected_corner = np.maximum.reduce([u[:8, :8] for u in uniques])
        assert np.array_equal(out[:8, :8], expected_corner)

        # macOS reports ru_maxrss in bytes. Inputs are ~1.44 GB; the fold
        # itself must add no more than a few working buffers.
        growth = rss_after - rss_before
        assert growth < 6 * 1024**3, f"fold grew RSS by {growth / 1e9:.2f} GB"
        # Report timing for the Phase-1 acceptance record.
        print(f"\n24 MP x 20-image blend: {elapsed:.2f} s "
              f"(RSS growth {growth / 1e9:.2f} GB)")
        assert elapsed < 120.0
