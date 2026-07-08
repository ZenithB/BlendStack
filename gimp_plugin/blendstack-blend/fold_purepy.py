"""Pure-Python blend backend — no NumPy (project brief §6, mitigation 3).

GIMP's bundled Python on Apple Silicon macOS runs under a hardened runtime
that enforces **library validation**: it refuses to load any compiled
extension not signed by GIMP's own Team ID. NumPy's C extensions are
ad-hoc signed, so `import numpy` fails inside GIMP with a Library
Validation error — and there is no way to make NumPy load without
modifying GIMP's own code signature (a security-relevant change the user
must decide on). Brief §6 anticipates exactly this and sanctions a
"slow pure-Python fallback for the fold only" as the worst-case mitigation.

This module is that fallback. It has **zero dependencies** (stdlib only)
and reimplements the blend maths from brief §4.2 and the geometry/alpha
handling the plugin needs, operating on `array('f')` buffers of float32
that come straight from / go straight to GEGL:

* :func:`layer_to_canvas` — place a layer's float pixels on a black,
  canvas-sized background at its offset (intersection-clipped), flattening
  alpha against black (rgb x a), matching the core's load policy.
* :func:`fold` — left-fold the canvas layers with Canon Bright / Dark,
  honouring softness, bias and comparison basis, then clip to 0..1.
* :func:`to_bytes` — serialise the result for `Gegl.Buffer.set`.

Correctness: for the default case (softness 0, bias 0, per-channel basis)
the per-channel `max`/`min` selection is **bit-identical** to NumPy's
`np.maximum` / `np.minimum` on the same float32 values (verified against
`blendstack.core.engine.fold_images` in test_plugin_logic.py). The soft /
biased / luminance paths compute in float64 and store to float32, so they
match the NumPy engine to within ~1 float32 ULP rather than bit-exactly.

Metadata constants (`MIN_LAYERS`, `MAX_LAYERS`, mode labels) live here too
so the plugin can register and validate without importing NumPy at all.
"""

from __future__ import annotations

import array
import math
from typing import Iterable, List, Sequence, Tuple

# Blend size limits mirror the engine (brief §2: 2-20 images).
MIN_LAYERS = 2
MAX_LAYERS = 20

# Fixed v1 mode registry (name -> label). Kept NumPy-free on purpose.
_MODES = (("canon_bright", "Canon Bright"), ("canon_dark", "Canon Dark"))

_LUMA = (0.2126, 0.7152, 0.0722)          # Rec.709 luma (brief §4.1/§4.2)
_BIAS_SCALE = 0.25 / 100.0                # UI -100..100 -> -0.25..0.25
_SOFT_SCALE = 0.25 / 100.0                # UI 1..100    -> t = softness/400

Canvas = "array.array"  # length canvas_w*canvas_h*3, float32, RGB interleaved


def mode_choices() -> List[Tuple[str, str]]:
    """(name, label) pairs for the two v1 modes."""
    return list(_MODES)


def mode_label(mode: str) -> str:
    for name, label in _MODES:
        if name == mode:
            return label
    return mode


# --------------------------------------------------------------------------
# Geometry + alpha: layer float bytes -> canvas-sized array('f') on black
# --------------------------------------------------------------------------

def layer_to_canvas(
    data: bytes,
    layer_w: int,
    layer_h: int,
    has_alpha: bool,
    off_x: int,
    off_y: int,
    canvas_w: int,
    canvas_h: int,
) -> array.array:
    """Place a layer's float32 pixels on a black canvas-sized background.

    ``data`` is the raw little-endian float32 buffer from ``Gegl.Buffer.get``
    in ``R'G'B' float`` (3 comps) or ``R'G'B'A float`` (4 comps). Regions
    outside the layer stay black; alpha is flattened against black (rgb x a),
    matching :mod:`blendstack.core.io`.
    """
    src = array.array("f")
    src.frombytes(data)
    canvas = array.array("f", bytes(4 * canvas_w * canvas_h * 3))  # zeros

    ncomp = 4 if has_alpha else 3
    x0 = max(0, off_x)
    x1 = min(canvas_w, off_x + layer_w)
    y0 = max(0, off_y)
    y1 = min(canvas_h, off_y + layer_h)
    if x1 <= x0 or y1 <= y0:
        return canvas  # layer entirely off-canvas -> all black

    span = x1 - x0
    for y in range(y0, y1):
        ly = y - off_y
        lx0 = x0 - off_x
        dst = (y * canvas_w + x0) * 3
        if has_alpha:
            s = (ly * layer_w + lx0) * 4
            for i in range(span):
                a = src[s + 3]
                canvas[dst] = src[s] * a
                canvas[dst + 1] = src[s + 1] * a
                canvas[dst + 2] = src[s + 2] * a
                s += 4
                dst += 3
        else:
            s = (ly * layer_w + lx0) * 3
            canvas[dst:dst + span * 3] = src[s:s + span * 3]
    return canvas


# --------------------------------------------------------------------------
# Fold (brief §4.2)
# --------------------------------------------------------------------------

def fold(
    canvases: Sequence[array.array],
    mode: str,
    softness: float,
    bias_ui: float,
    basis: str,
) -> array.array:
    """Left-fold canvas layers into one composite, then clip to 0..1.

    ``canvases`` are equal-length ``array('f')`` buffers (RGB interleaved);
    the first is the base (its opacity is ignored, per brief §4). Mirrors
    :func:`blendstack.core.engine.fold_images` at opacity 100 for all layers.
    """
    if basis not in ("per_channel", "luminance"):
        raise ValueError(f"Unknown comparison basis '{basis}'")
    direction = 1 if mode == "canon_bright" else -1
    bias = float(bias_ui) * _BIAS_SCALE

    acc = canvases[0]
    for inc in canvases[1:]:
        acc = _blend_pair(acc, inc, direction, float(softness), bias, basis)

    # Final clip to 0..1 (brief §4 pipeline). One pass; returns float32.
    return array.array("f", (0.0 if v < 0.0 else 1.0 if v > 1.0 else v for v in acc))


def _blend_pair(acc, inc, direction, softness, bias, basis):
    # Fast path: default settings, per-channel — bit-exact vs np.maximum/minimum.
    if softness <= 0.0 and bias == 0.0 and basis == "per_channel":
        picker = max if direction > 0 else min
        return array.array("f", map(picker, acc, inc))

    out = array.array("f", bytes(len(acc) * 4))
    hard = softness <= 0.0
    t = softness * _SOFT_SCALE if not hard else 0.0
    lr, lg, lb = _LUMA
    n = len(acc)

    if basis == "luminance":
        # One selection weight per pixel, applied to all three channels.
        for p in range(0, n, 3):
            a0, a1, a2 = acc[p], acc[p + 1], acc[p + 2]
            b0, b1, b2 = inc[p], inc[p + 1], inc[p + 2]
            a_cmp = lr * a0 + lg * a1 + lb * a2
            b_cmp = lr * b0 + lg * b1 + lb * b2
            d = (b_cmp - a_cmp if direction > 0 else a_cmp - b_cmp) + bias
            if hard:
                if d > 0.0:
                    out[p], out[p + 1], out[p + 2] = b0, b1, b2
                else:
                    out[p], out[p + 1], out[p + 2] = a0, a1, a2
            else:
                w = _sigmoid(d / t)
                out[p] = a0 + (b0 - a0) * w
                out[p + 1] = a1 + (b1 - a1) * w
                out[p + 2] = a2 + (b2 - a2) * w
    else:
        # Per-channel: three independent selections (colours may mix).
        for i in range(n):
            a = acc[i]
            b = inc[i]
            d = (b - a if direction > 0 else a - b) + bias
            if hard:
                out[i] = b if d > 0.0 else a
            else:
                w = _sigmoid(d / t)
                out[i] = a + (b - a) * w
    return out


def _sigmoid(z: float) -> float:
    if z < -60.0:
        z = -60.0
    elif z > 60.0:
        z = 60.0
    return 1.0 / (1.0 + math.exp(-z))


def to_bytes(canvas: array.array) -> bytes:
    """Serialise a canvas array back to a little-endian float32 buffer."""
    return canvas.tobytes()
