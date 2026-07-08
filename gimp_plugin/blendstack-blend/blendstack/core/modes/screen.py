"""Screen blend mode.

``out = 1 - (1 - A) * (1 - B)`` per channel. A "soft lighten": brightens
without ever pushing past white, so stacked light streaks / bright ICM
movement build up luminously instead of blowing out (unlike Additive).

Computed in gamma space to match the familiar GIMP / Photoshop Screen look
(``needs_linear = False``). Screen is commutative and associative, so the
fold is order-independent — like the Canon comparative modes.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, register_mode


@register_mode
class Screen(BlendMode):
    """Screen — 1 - (1-A)(1-B), self-limiting brighten."""

    name: ClassVar[str] = "screen"
    label: ClassVar[str] = "Screen"
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
        return 1.0 - (1.0 - accumulator) * (1.0 - incoming)
