"""Overlay blend mode (GIMP / Photoshop style).

Per channel, with A = accumulator (the base) and B = incoming:

    out = 2 * A * B                    where A < 0.5   (multiply-like shadows)
    out = 1 - 2 * (1 - A) * (1 - B)    where A >= 0.5  (screen-like highlights)

A contrast-boosting composite: dark areas of the base darken, light areas
lighten, so a movement/texture frame laid over a base comes through punchy
and saturated. Closed on 0..1 (no clamping needed for in-range inputs).

Computed in gamma space to match GIMP / Photoshop Overlay (``needs_linear =
False``). Because the base A drives which branch is taken, Overlay is **not**
commutative — image order matters (top image = base), so the reorderable
strip is meaningful with this mode.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, register_mode


@register_mode
class Overlay(BlendMode):
    """Overlay — contrast-boosting multiply/screen composite (order matters)."""

    name: ClassVar[str] = "overlay"
    label: ClassVar[str] = "Overlay"
    needs_linear: ClassVar[bool] = False
    parameters: ClassVar[tuple] = ()

    def blend(
        self,
        accumulator: np.ndarray,
        incoming: np.ndarray,
        params: Mapping[str, Any] | None = None,
        count: int = 1,
    ) -> np.ndarray:
        del count
        a, b = accumulator, incoming
        return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
