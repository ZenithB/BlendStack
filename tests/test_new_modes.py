"""Tests for the post-v1 ICM blend modes: average, screen, multiply,
grain_merge, overlay (and the fold's count / pick_params contract)."""

from __future__ import annotations

import numpy as np
import pytest

from blendstack.core import engine
from blendstack.core.adjustments import linear_to_srgb, srgb_to_linear
from blendstack.core.modes import get_mode, mode_names


def _imgs(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((20, 16, 3)).astype(np.float32) for _ in range(n)]


def _fold_ref(op, imgs):
    acc = imgs[0].astype(np.float32).copy()
    for b in imgs[1:]:
        acc = op(acc, b)
    return np.clip(acc, 0.0, 1.0)


def test_all_modes_registered():
    for name in ("average", "screen", "multiply", "grain_merge", "overlay"):
        assert name in mode_names()
    # No mode declares softness/bias/basis except the two Canon modes.
    for name in ("average", "screen", "multiply", "grain_merge", "overlay"):
        assert get_mode(name).parameters == ()


def test_needs_linear_flags():
    assert get_mode("average").needs_linear is True
    for name in ("screen", "multiply", "grain_merge", "overlay"):
        assert get_mode(name).needs_linear is False


def test_average_is_linear_light_mean():
    imgs = _imgs()
    got = engine.blend_arrays(imgs, mode="average")
    expect = linear_to_srgb(np.mean([srgb_to_linear(i) for i in imgs], axis=0))
    assert np.allclose(got, expect, atol=1e-6)


def test_average_order_independent():
    imgs = _imgs()
    a = engine.blend_arrays(imgs, mode="average")
    b = engine.blend_arrays([imgs[i] for i in (3, 0, 4, 1, 2)], mode="average")
    assert np.allclose(a, b, atol=1e-6)


def test_average_two_images_is_midpoint_in_linear():
    a = np.full((4, 4, 3), 0.2, dtype=np.float32)
    b = np.full((4, 4, 3), 0.8, dtype=np.float32)
    got = engine.blend_arrays([a, b], mode="average")
    expect = linear_to_srgb(0.5 * (srgb_to_linear(a) + srgb_to_linear(b)))
    assert np.allclose(got, expect, atol=1e-6)


def test_screen_formula_and_order_independent():
    imgs = _imgs()
    got = engine.blend_arrays(imgs, mode="screen")
    ref = _fold_ref(lambda a, b: 1.0 - (1.0 - a) * (1.0 - b), imgs)
    assert np.allclose(got, ref, atol=1e-6)
    assert got.min() >= 0.0 and got.max() <= 1.0
    shuf = engine.blend_arrays([imgs[i] for i in (4, 3, 2, 1, 0)], mode="screen")
    assert np.allclose(got, shuf, atol=1e-6)


def test_multiply_formula_and_order_independent():
    imgs = _imgs()
    got = engine.blend_arrays(imgs, mode="multiply")
    ref = _fold_ref(lambda a, b: a * b, imgs)
    assert np.allclose(got, ref, atol=1e-6)
    shuf = engine.blend_arrays([imgs[i] for i in (2, 4, 0, 3, 1)], mode="multiply")
    assert np.allclose(got, shuf, atol=1e-6)


def test_grain_merge_formula_clamped():
    imgs = _imgs()
    got = engine.blend_arrays(imgs, mode="grain_merge")
    ref = _fold_ref(lambda a, b: np.clip(a + b - 0.5, 0.0, 1.0), imgs)
    assert np.allclose(got, ref, atol=1e-6)
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_overlay_formula_and_order_dependent():
    imgs = _imgs()
    got = engine.blend_arrays(imgs, mode="overlay")
    ref = _fold_ref(
        lambda a, b: np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)), imgs
    )
    assert np.allclose(got, ref, atol=1e-6)
    assert got.min() >= 0.0 and got.max() <= 1.0
    shuf = engine.blend_arrays([imgs[i] for i in (4, 3, 2, 1, 0)], mode="overlay")
    assert not np.allclose(got, shuf, atol=1e-4)  # base drives the branch


def test_new_modes_ignore_foreign_params():
    # Frontends carry a global softness/bias/basis dict; param-free modes
    # must ignore it rather than raise (pick_params leniency).
    imgs = _imgs()
    plain = engine.blend_arrays(imgs, mode="multiply")
    with_extra = engine.blend_arrays(
        imgs, mode="multiply",
        params={"softness": 50, "bias": 20, "basis": "luminance"},
    )
    assert np.allclose(plain, with_extra, atol=1e-6)


def test_comparative_modes_unchanged():
    imgs = _imgs()
    assert np.array_equal(
        engine.blend_arrays(imgs, mode="canon_bright"),
        _fold_ref(np.maximum, imgs),
    )
    assert np.array_equal(
        engine.blend_arrays(imgs, mode="canon_dark"),
        _fold_ref(np.minimum, imgs),
    )
