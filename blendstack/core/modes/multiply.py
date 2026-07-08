"""Multiply blend mode.

``out = A * B`` per channel. Darkens: wherever any frame is dark the result
is dark, so density builds up — good for moody skies and combining dark ICM
movement. Like stacking transparencies.

Computed in gamma space to match GIMP / Photoshop Multiply
(``needs_linear = False``). Multiply is commutative and associative, so the
fold is order-independent.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, register_mode


@register_mode
class Multiply(BlendMode):
    """Multiply — A * B, builds density."""

    name: ClassVar[str] = "multiply"
    label: ClassVar[str] = "Multiply"
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
        return accumulator * incoming
