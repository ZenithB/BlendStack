"""Size-mismatch handling: scale-to-smallest, cover-crop, Lanczos (brief §4.3).

Target dimensions = the smallest input **by area**.  Each other image is
scaled by ``max(target_w / w, target_h / h)`` (aspect-preserving *cover*
scaling — no distortion, edge content may be lost), resampled with Lanczos,
then centre-cropped to the target.  Identical-size inputs pass through
untouched (same array object, no copy).

v1.1 placement: :func:`cover_scale` scales without cropping, :func:`place`
puts the scaled layer on the target canvas (centred, then shifted by an
integer pixel offset) and returns a coverage mask so the engine can treat
the uncovered gap as transparent.  ``place(cover_scale(img, t), t, 0, 0)``
equals :func:`conform` bit-exactly.

Pillow is used only as a resampling kernel (float32 "F"-mode channels);
no UI toolkits are imported.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "target_dimensions",
    "conform",
    "conform_stack",
    "cover_scale",
    "place",
    "offset_px",
]

Size = Tuple[int, int]  # (width, height)


def target_dimensions(sizes: Sequence[Size]) -> Size:
    """Return the (width, height) of the smallest input by area.

    Ties resolve to the earliest image in the list.
    """
    if not sizes:
        raise ValueError("target_dimensions() requires at least one size")
    return min(sizes, key=lambda wh: wh[0] * wh[1])


def _resize_lanczos(image: np.ndarray, size: Size) -> np.ndarray:
    """Lanczos-resample an (H, W, 3) float32 array to (width, height)."""
    width, height = size
    channels = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(image[..., c]), mode="F").resize(
                (width, height), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def conform(image: np.ndarray, target: Size) -> np.ndarray:
    """Cover-scale + centre-crop one image to ``target`` (width, height).

    An image already at the target size is returned untouched (no copy).
    The result is clipped to 0–1 because Lanczos ringing can overshoot.
    """
    height, width = image.shape[:2]
    target_w, target_h = target
    if (width, height) == (target_w, target_h):
        return image

    scale = max(target_w / width, target_h / height)
    # Round to nearest but never below the target (cover must fully cover).
    new_w = max(target_w, int(round(width * scale)))
    new_h = max(target_h, int(round(height * scale)))
    resized = _resize_lanczos(image, (new_w, new_h))

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = resized[top : top + target_h, left : left + target_w]
    return np.clip(cropped, 0.0, 1.0).astype(np.float32, copy=False)


def conform_stack(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Conform every image to the smallest input's dimensions (by area)."""
    sizes: list[Size] = [(im.shape[1], im.shape[0]) for im in images]
    target = target_dimensions(sizes)
    return [conform(im, target) for im in images]


def cover_scale(image: np.ndarray, target: Size) -> np.ndarray:
    """Scale ``image`` (Lanczos, like :func:`conform`) by
    ``max(tw / w, th / h)`` so it *covers* ``target`` (width, height), but do
    NOT crop.  Each dimension is rounded to nearest and never below the
    target.  The result is clipped to 0–1 (Lanczos ringing).  An image
    already at the target size is returned untouched (the same object)."""
    height, width = image.shape[:2]
    target_w, target_h = target
    if (width, height) == (target_w, target_h):
        return image
    scale = max(target_w / width, target_h / height)
    new_w = max(target_w, int(round(width * scale)))
    new_h = max(target_h, int(round(height * scale)))
    resized = _resize_lanczos(image, (new_w, new_h))
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)


def place(
    scaled: np.ndarray, target: Size, dx_px: float = 0, dy_px: float = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Put ``scaled`` on a ``(th, tw, 3)`` canvas.

    The image is first centred (exactly the centre-crop :func:`conform`
    does), then shifted by ``round(dx_px)`` / ``round(dy_px)`` whole pixels
    (+x right, +y down).  Overscan beyond the canvas edge can slide back
    into view when the layer is moved.

    Returns ``(canvas, mask)``: ``canvas`` is float32, zeros where the layer
    does not cover; ``mask`` is float32 ``(th, tw, 1)`` with 1.0 where
    covered and 0.0 elsewhere, or ``None`` when the whole canvas is covered
    (so the no-move case costs nothing).  An already-target-size image with
    no offset is returned as the same object."""
    target_w, target_h = target
    sh, sw = scaled.shape[:2]
    dx = int(round(dx_px))
    dy = int(round(dy_px))
    # canvas (cx, cy) shows source (cx + sx0, cy + sy0)
    sx0 = (sw - target_w) // 2 - dx
    sy0 = (sh - target_h) // 2 - dy
    cx0, cx1 = max(0, -sx0), min(target_w, sw - sx0)
    cy0, cy1 = max(0, -sy0), min(target_h, sh - sy0)

    if cx0 == 0 and cy0 == 0 and cx1 == target_w and cy1 == target_h:
        if (sw, sh) == (target_w, target_h) and sx0 == 0 and sy0 == 0:
            return scaled.astype(np.float32, copy=False), None
        view = scaled[sy0 : sy0 + target_h, sx0 : sx0 + target_w]
        return np.ascontiguousarray(view, dtype=np.float32), None

    canvas = np.zeros((target_h, target_w, scaled.shape[2]), dtype=np.float32)
    mask = np.zeros((target_h, target_w, 1), dtype=np.float32)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = scaled[
            cy0 + sy0 : cy1 + sy0, cx0 + sx0 : cx1 + sx0
        ]
        mask[cy0:cy1, cx0:cx1] = 1.0
    return canvas, mask


def offset_px(move_x: float, move_y: float, target: Size) -> Tuple[int, int]:
    """Convert placement fractions (of canvas width / height, +x right,
    +y down) to whole-pixel offsets: ``(round(move_x * tw), round(move_y * th))``."""
    target_w, target_h = target
    return int(round(move_x * target_w)), int(round(move_y * target_h))
