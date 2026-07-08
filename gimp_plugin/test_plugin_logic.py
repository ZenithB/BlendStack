#!/usr/bin/env python3
"""Tests for the GIMP plugin's numeric logic — no GIMP required.

Run with the repo virtualenv:

    /Users/chris/Documents/BlendStack/.venv/bin/python gimp_plugin/test_plugin_logic.py

(also collectable by pytest).  Prerequisite: ``python3 gimp_plugin/sync_core.py``
must have vendored the core into the plugin folder — the tests import
``blend_logic`` and the *vendored* ``blendstack`` package exactly as the
plugin does inside GIMP, then compare against ``blendstack.core.engine``
(same package; engine is importable here because this interpreter has
Pillow).

Covers:
* fold_visible is bit-exact against engine.fold_images for both modes,
  both bases, several softness/bias settings;
* composite_at_canvas: black background, offsets (incl. negative),
  out-of-canvas clipping, exact-fit fast path;
* flatten_alpha matches the core policy (rgb * a);
* uint8- / uint16- / float-sourced inputs pick identical winners at
  default settings (8/16/32-bit images behave identically);
* layer-count limits raise clear errors;
* the actual plugin file parses (ast) and has shebang + exec bit.
"""

from __future__ import annotations

import ast
import os
import stat
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(REPO_ROOT, "gimp_plugin", "blendstack-blend")
PLUGIN_FILE = os.path.join(PLUGIN_DIR, "blendstack-blend.py")

VENDORED_CORE = os.path.join(PLUGIN_DIR, "blendstack", "core")
assert os.path.isdir(VENDORED_CORE), (
    "Vendored core missing — run: python3 gimp_plugin/sync_core.py"
)

# Import blend_logic and blendstack exactly as the plugin does: from the
# plugin folder (the vendored copy), ahead of the repo root.
sys.path.insert(0, PLUGIN_DIR)

import numpy as np  # noqa: E402

import blend_logic  # noqa: E402
from blendstack.core import engine  # noqa: E402  (vendored copy; needs Pillow)

assert engine.__file__ and os.path.realpath(engine.__file__).startswith(
    os.path.realpath(PLUGIN_DIR)
), f"engine resolved outside the vendored copy: {engine.__file__}"


def _random_stack(rng, n=5, h=37, w=53):
    return [rng.random((h, w, 3), dtype=np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Fold parity with the frozen engine
# ---------------------------------------------------------------------------

def test_fold_matches_engine_bit_exact():
    rng = np.random.default_rng(42)
    images = _random_stack(rng)
    settings = [
        (0.0, 0.0),      # defaults — pixel-exact Canon
        (0.0, 37.5),     # hard select with bias
        (12.5, -10.0),
        (60.0, -40.0),
        (100.0, 100.0),  # extremes
    ]
    for mode in engine.mode_names():
        for basis in ("per_channel", "luminance"):
            for softness, bias in settings:
                params = {"softness": softness, "bias": bias, "basis": basis}
                expected = engine.fold_images(images, mode, params)
                got = blend_logic.fold_visible(images, mode, softness, bias, basis)
                assert got.dtype == np.float32
                assert np.array_equal(got, expected), (
                    f"fold mismatch: {mode} {params}"
                )


def test_fold_defaults_equal_hard_max_min():
    rng = np.random.default_rng(7)
    images = _random_stack(rng, n=4)
    stack = np.stack(images)
    bright = blend_logic.fold_visible(images, "canon_bright", 0.0, 0.0, "per_channel")
    dark = blend_logic.fold_visible(images, "canon_dark", 0.0, 0.0, "per_channel")
    assert np.array_equal(bright, np.max(stack, axis=0))
    assert np.array_equal(dark, np.min(stack, axis=0))


# ---------------------------------------------------------------------------
# 2. Canvas compositing (black background, offsets, clipping)
# ---------------------------------------------------------------------------

def test_composite_smaller_layer_with_offset():
    layer = np.ones((4, 3, 3), dtype=np.float32) * 0.5
    out = blend_logic.composite_at_canvas((8, 10), layer, (2, 3))  # (w=8, h=10)
    assert out.shape == (10, 8, 3)
    assert np.all(out[3:7, 2:5] == 0.5)          # layer pixels in place
    mask = np.zeros((10, 8), dtype=bool)
    mask[3:7, 2:5] = True
    assert np.all(out[~mask] == 0.0)             # black everywhere else


def test_composite_negative_offsets_clip():
    layer = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    out = blend_logic.composite_at_canvas((8, 10), layer, (-2, -1))
    assert out.shape == (10, 8, 3)
    # Only layer columns >= 2 and rows >= 1 survive, placed at canvas origin.
    assert np.array_equal(out[0:3, 0:1], layer[1:4, 2:3])
    assert np.all(out[3:, :] == 0.0) and np.all(out[:, 1:] == 0.0)


def test_composite_layer_larger_than_canvas():
    layer = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
    out = blend_logic.composite_at_canvas((4, 3), layer, (-1, -2))
    assert out.shape == (3, 4, 3)
    assert np.array_equal(out, layer[2:5, 1:5])


def test_composite_fully_outside_canvas_is_black():
    layer = np.ones((4, 4, 3), dtype=np.float32)
    for offsets in [(100, 100), (-4, 0), (0, -4), (8, 0), (0, 10)]:
        out = blend_logic.composite_at_canvas((8, 10), layer, offsets)
        assert out.shape == (10, 8, 3)
        assert np.all(out == 0.0), f"offsets {offsets} leaked pixels"


def test_composite_exact_fit_passthrough():
    rng = np.random.default_rng(1)
    layer = rng.random((10, 8, 3), dtype=np.float32)
    out = blend_logic.composite_at_canvas((8, 10), layer, (0, 0))
    assert np.array_equal(out, layer)


# ---------------------------------------------------------------------------
# 3. Alpha flattening matches core policy (rgb * a against black)
# ---------------------------------------------------------------------------

def test_flatten_alpha_matches_core_policy():
    rng = np.random.default_rng(3)
    rgba = rng.random((9, 11, 4), dtype=np.float32)
    flat = blend_logic.flatten_alpha(rgba)
    assert flat.shape == (9, 11, 3) and flat.dtype == np.float32
    assert np.array_equal(flat, rgba[..., :3] * rgba[..., 3:4])
    # alpha = 0 -> black; alpha = 1 -> untouched rgb
    rgba[..., 3] = 0.0
    assert np.all(blend_logic.flatten_alpha(rgba) == 0.0)
    rgba[..., 3] = 1.0
    assert np.array_equal(blend_logic.flatten_alpha(rgba), rgba[..., :3])


# ---------------------------------------------------------------------------
# 4. Bit-depth equivalence: identical winners at default settings
# ---------------------------------------------------------------------------

def test_bit_depths_pick_identical_winners():
    rng = np.random.default_rng(11)
    u8_images = [rng.integers(0, 256, size=(23, 31, 3), dtype=np.uint8)
                 for _ in range(4)]
    # The three float representations GEGL would hand us for 8-, 16- and
    # 32-bit float precision sources of the same picture.
    reps = {
        "8-bit": [(u8 / 255.0).astype(np.float32) for u8 in u8_images],
        "16-bit": [((u8.astype(np.uint16) * 257) / 65535.0).astype(np.float32)
                   for u8 in u8_images],
        "float": [u8.astype(np.float32) / np.float32(255.0)
                  for u8 in u8_images],
    }
    for mode, reduce_fn, arg_fn in [
        ("canon_bright", np.max, np.argmax),
        ("canon_dark", np.min, np.argmin),
    ]:
        winners = {}
        for name, images in reps.items():
            stack = np.stack(images)
            out = blend_logic.fold_visible(images, mode, 0.0, 0.0, "per_channel")
            # Defaults = exact per-channel max/min of that representation…
            assert np.array_equal(out, reduce_fn(stack, axis=0)), (mode, name)
            # …and the winning source index per channel is depth-independent.
            winners[name] = arg_fn(stack, axis=0)
        first = winners["8-bit"]
        for name, w in winners.items():
            assert np.array_equal(w, first), (
                f"{mode}: {name} picked different winners than 8-bit"
            )


# ---------------------------------------------------------------------------
# 5. Layer-count limits
# ---------------------------------------------------------------------------

def test_layer_count_limits():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    for bad in ([], [img], [img] * 21):
        try:
            blend_logic.fold_visible(bad, "canon_bright", 0.0, 0.0, "per_channel")
        except ValueError as exc:
            n = len(bad)
            assert str(blend_logic.MAX_LAYERS if n > 20 else blend_logic.MIN_LAYERS) \
                in str(exc)
        else:
            raise AssertionError(f"{len(bad)} layers should have been rejected")
    # 2 and 20 are accepted.
    blend_logic.fold_visible([img] * 2, "canon_bright", 0.0, 0.0, "per_channel")
    blend_logic.fold_visible([img] * 20, "canon_dark", 0.0, 0.0, "per_channel")


# ---------------------------------------------------------------------------
# 6. The plugin file itself: parses, shebang, executable
# ---------------------------------------------------------------------------

def test_plugin_file_parses_and_is_executable():
    with open(PLUGIN_FILE, encoding="utf-8") as fh:
        source = fh.read()
    ast.parse(source)  # SyntaxError if the GIMP-side file is broken
    assert source.startswith("#!/usr/bin/env python3")
    mode = os.stat(PLUGIN_FILE).st_mode
    assert mode & stat.S_IXUSR, "plugin file must be chmod +x for GIMP"
    # The plugin must never import the PIL-dependent engine module.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "engine" not in node.module and "core.io" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "engine" not in alias.name


# ---------------------------------------------------------------------------
# 7. Pure-Python fallback backend (fold_purepy) — used inside GIMP when
#    NumPy cannot load. Must match the NumPy engine.
# ---------------------------------------------------------------------------

import array  # noqa: E402
import fold_purepy as pp  # noqa: E402


def _pp_fold(imgs, mode, soft, bias, basis):
    h, w = imgs[0].shape[:2]
    canvases = [array.array("f", im.astype(np.float32).tobytes()) for im in imgs]
    out = pp.fold(canvases, mode, soft, bias, basis)
    return np.frombuffer(out.tobytes(), dtype=np.float32).reshape(h, w, 3)


def test_purepy_fold_defaults_bit_exact_vs_engine():
    rng = np.random.default_rng(11)
    imgs = [rng.random((24, 18, 3)).astype(np.float32) for _ in range(5)]
    for mode in ("canon_bright", "canon_dark"):
        eng = engine.fold_images(
            imgs, mode=mode,
            params={"softness": 0, "bias": 0, "basis": "per_channel"},
        )
        got = _pp_fold(imgs, mode, 0, 0, "per_channel")
        assert np.array_equal(eng, got), f"{mode} purepy not bit-exact"


def test_purepy_fold_soft_bias_luma_close_to_engine():
    rng = np.random.default_rng(12)
    imgs = [rng.random((24, 18, 3)).astype(np.float32) for _ in range(4)]
    cases = [
        ("canon_bright", 15, 0, "per_channel"),
        ("canon_dark", 40, -20, "per_channel"),
        ("canon_bright", 0, 30, "per_channel"),
        ("canon_bright", 0, 0, "luminance"),
        ("canon_dark", 25, 10, "luminance"),
    ]
    for mode, s, b, basis in cases:
        eng = engine.fold_images(
            imgs, mode=mode, params={"softness": s, "bias": b, "basis": basis}
        )
        got = _pp_fold(imgs, mode, s, b, basis)
        md = float(np.max(np.abs(eng - got)))
        assert md < 1e-5, f"{mode} s={s} b={b} {basis}: diff {md}"


def test_purepy_layer_to_canvas_matches_numpy_backend():
    rng = np.random.default_rng(13)
    cw, ch = 20, 16
    cases = [
        ((10, 8), (5, 4), False),
        ((12, 10), (-3, -2), False),
        ((30, 25), (-4, -5), False),
        ((14, 12), (12, 9), False),
        ((6, 6), (40, 40), False),
        ((cw, ch), (0, 0), False),
        ((10, 8), (3, 3), True),
        ((12, 10), (-4, -3), True),
    ]
    for (lw, lh), (ox, oy), alpha in cases:
        nc = 4 if alpha else 3
        layer = rng.random((lh, lw, nc)).astype(np.float32)
        canv = pp.layer_to_canvas(
            layer.tobytes(), lw, lh, alpha, ox, oy, cw, ch
        )
        got = np.frombuffer(canv.tobytes(), dtype=np.float32).reshape(ch, cw, 3)
        rgb = blend_logic.flatten_alpha(layer) if alpha else layer
        exp = blend_logic.composite_at_canvas((cw, ch), rgb, (ox, oy))
        assert np.array_equal(got, exp), f"purepy composite mismatch {(lw, lh, ox, oy, alpha)}"


def test_purepy_metadata_matches():
    assert pp.MIN_LAYERS == blend_logic.MIN_LAYERS
    assert pp.MAX_LAYERS == blend_logic.MAX_LAYERS
    assert pp.mode_choices() == blend_logic.mode_choices()
    assert pp.mode_label("canon_bright") == "Canon Bright"


def main() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
