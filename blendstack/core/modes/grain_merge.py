"""Grain Merge blend mode (GIMP-style).

``out = A + B - 0.5`` per channel, clamped to 0..1. A midtone-neutral add:
0.5 is the neutral point, so one frame's texture / movement is overlaid onto
another without shifting overall brightness. Useful for layering ICM streak
detail.

Computed in gamma space to match GIMP's Grain Merge (``needs_linear =
False``). The output is clamped per fold step (as GIMP clamps per layer), so
chaining many layers stays well-behaved; because of that clamping the fold
is order-independent only while values stay in range.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, register_mode


@register_mode
class GrainMerge(BlendMode):
    """Grain Merge — A + B - 0.5, midtone-neutral texture add."""

    name: ClassVar[str] = "grain_merge"
    label: ClassVar[str] = "Grain Merge"
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
        return np.clip(accumulator + incoming - 0.5, 0.0, 1.0)
