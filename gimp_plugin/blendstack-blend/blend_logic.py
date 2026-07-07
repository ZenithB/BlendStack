"""GIMP-independent blend logic for the BlendStack GIMP plugin.

Everything in this module is plain NumPy + the BlendStack core — no ``gi``
imports — so the whole numeric path can be unit-tested outside GIMP
(see ``gimp_plugin/test_plugin_logic.py``, which verifies it bit-exact
against ``blendstack.core.engine.fold_images``).

Import-chain decision (why this module does NOT import
``blendstack.core.engine``):

* ``blendstack/__init__.py`` and ``blendstack/core/__init__.py`` are
  docstring-only — importing them pulls in nothing.
* ``blendstack.core.modes`` (registry + both Canon modes) and
  ``blendstack.core.adjustments`` import only NumPy — safe inside GIMP.
* ``blendstack.core.engine`` imports ``blendstack.core.io`` and
  ``blendstack.core.geometry`` at module top, and **both import PIL
  (Pillow)**, which GIMP's bundled Python does not ship.  Importing the
  engine inside GIMP would therefore fail even with NumPy installed.

So the plugin imports the mode registry directly and re-implements the
tiny fold loop here with the exact semantics of ``engine.fold_images``
(brief §4):

    accumulator = images[0]                      # first image = base
    for each subsequent image:
        blended     = mode.blend(accumulator, image, params)
        accumulator = lerp(accumulator, blended, opacity)
    clip accumulator to 0..1

The plugin uses opacity 100 % for every layer, so the lerp step is the
identity and the fold reduces to successive ``mode.blend`` calls.  The
engine's linear-light scaffolding (``mode.needs_linear``) is mirrored via
``blendstack.core.adjustments`` so a future linear mode dropped into the
registry keeps working here unchanged.  Core is frozen — nothing in
``blendstack/`` is modified, only imported.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from blendstack.core import adjustments as _adj
from blendstack.core.modes import get_mode, mode_names

__all__ = [
    "MIN_LAYERS",
    "MAX_LAYERS",
    "mode_choices",
    "mode_label",
    "flatten_alpha",
    "composite_at_canvas",
    "fold_visible",
]

#: Blend size limits — mirror ``engine.MIN_IMAGES`` / ``engine.MAX_IMAGES``
#: (brief §2: 2–20 input images per blend).  Duplicated here because the
#: engine module cannot be imported inside GIMP (see module docstring).
MIN_LAYERS = 2
MAX_LAYERS = 20


def mode_choices() -> list[tuple[str, str]]:
    """``[(name, ui_label), ...]`` for every registered blend mode, in
    registration order — e.g. ``[("canon_bright", "Canon Bright"), ...]``."""
    return [(name, get_mode(name).label) for name in mode_names()]


def mode_label(name: str) -> str:
    """UI label for a registered mode name (``ValueError`` if unknown)."""
    return get_mode(name).label


def flatten_alpha(rgba: np.ndarray) -> np.ndarray:
    """Flatten an (H, W, 4) float RGBA array against black: ``rgb * a``.

    Matches the core's alpha policy (brief §3 design rule 3: alpha channels
    are flattened against black on load).  Returns float32 (H, W, 3).
    """
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected an (H, W, 4) RGBA array, got shape {arr.shape}")
    return (arr[..., :3] * arr[..., 3:4]).astype(np.float32, copy=False)


def composite_at_canvas(
    canvas_wh: Tuple[int, int],
    layer_array: np.ndarray,
    offsets: Tuple[int, int],
) -> np.ndarray:
    """Place a layer's pixels on a black, canvas-sized background.

    Brief §6: layers offset or smaller than the canvas are composited
    against black at canvas size.  ``canvas_wh`` is ``(width, height)``,
    ``layer_array`` is float RGB (h, w, 3), ``offsets`` is the layer's
    ``(offset_x, offset_y)`` relative to the canvas origin (may be
    negative).  The layer rectangle is intersected with the canvas
    rectangle; anything outside is discarded.  Returns float32
    (canvas_h, canvas_w, 3).
    """
    canvas_w, canvas_h = int(canvas_wh[0]), int(canvas_wh[1])
    arr = np.asarray(layer_array, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {arr.shape}")
    layer_h, layer_w = arr.shape[:2]
    off_x, off_y = int(offsets[0]), int(offsets[1])

    # Fast path: the layer exactly covers the canvas.
    if (off_x, off_y) == (0, 0) and (layer_w, layer_h) == (canvas_w, canvas_h):
        return arr

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    # Intersect the layer rectangle with the canvas rectangle.
    x0, y0 = max(off_x, 0), max(off_y, 0)
    x1, y1 = min(off_x + layer_w, canvas_w), min(off_y + layer_h, canvas_h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = arr[y0 - off_y : y1 - off_y, x0 - off_x : x1 - off_x]
    return canvas


def fold_visible(
    layers: Sequence[np.ndarray],
    mode: str,
    softness: float,
    bias: float,
    basis: str,
) -> np.ndarray:
    """Fold canvas-sized float RGB layers into one composite.

    ``layers`` must be ordered top-first (GIMP stacking order): the top
    layer is the first/base image of the fold, exactly as brief §6
    specifies.  All arrays must share one (H, W, 3) shape — run
    :func:`composite_at_canvas` first.

    Semantics are identical to ``engine.fold_images(layers, mode, params)``
    with all opacities at 100 % (verified bit-exact by the test suite):
    first image = accumulator, each subsequent image folded with
    ``mode.blend``, result clipped to 0..1, returned as float32.
    """
    n = len(layers)
    if n < MIN_LAYERS:
        raise ValueError(
            f"BlendStack needs at least {MIN_LAYERS} visible layers, got {n}"
        )
    if n > MAX_LAYERS:
        raise ValueError(
            f"BlendStack blends at most {MAX_LAYERS} layers (engine limit), "
            f"got {n} — hide some layers and re-run"
        )

    mode_obj = get_mode(mode)
    params = mode_obj.resolve_params(
        {"softness": float(softness), "bias": float(bias), "basis": str(basis)}
    )

    accumulator: np.ndarray | None = None
    for image in layers:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB arrays, got shape {arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        if mode_obj.needs_linear:  # engine scaffolding, mirrored (brief §3)
            arr = _adj.srgb_to_linear(arr)
        if accumulator is None:
            accumulator = arr  # first image: base, opacity ignored (brief §4)
        else:
            if accumulator.shape != arr.shape:
                raise ValueError(
                    f"Layer shape {arr.shape} does not match {accumulator.shape}; "
                    "composite all layers to canvas size first"
                )
            # Opacity is 100 % for every layer in the plugin, so the
            # engine's lerp(accumulator, blended, opacity) is the identity.
            accumulator = mode_obj.blend(accumulator, arr, params)

    assert accumulator is not None
    if mode_obj.needs_linear:
        accumulator = _adj.linear_to_srgb(accumulator)
    return np.clip(accumulator, 0.0, 1.0).astype(np.float32, copy=False)
