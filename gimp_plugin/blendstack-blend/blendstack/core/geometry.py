"""Size-mismatch handling: scale-to-smallest, cover-crop, Lanczos (brief §4.3).

Target dimensions = the smallest input **by area**.  Each other image is
scaled by ``max(target_w / w, target_h / h)`` (aspect-preserving *cover*
scaling — no distortion, edge content may be lost), resampled with Lanczos,
then centre-cropped to the target.  Identical-size inputs pass through
untouched (same array object, no copy).

Pillow is used only as a resampling kernel (float32 "F"-mode channels);
no UI toolkits are imported.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = ["target_dimensions", "conform", "conform_stack"]

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
