"""Average (Mean) blend mode.

Running mean of all N source images — the classic long-exposure / ICM
simulation: many movement frames merge into one smooth, low-noise result.

The mean is computed in **linear light** (``needs_linear = True``) so that
brightness averages the way real photons do; averaging gamma-encoded values
would muddy the midtones. The engine linearises each frame before the fold
and re-encodes the result.

The fold is a left fold, so ``blend`` receives ``count`` — the number of
frames already folded into the accumulator — and weights the incoming frame
by ``1 / (count + 1)`` to keep a true running mean:

    new_mean = acc * count/(count+1) + incoming * 1/(count+1)

which is order-independent (the mean does not depend on frame order). With
per-image opacity below 100 % the fold's opacity lerp turns it into a
weighted mean, which is a reasonable, documented departure.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, register_mode


@register_mode
class Average(BlendMode):
    """Mean of all frames (linear-light)."""

    name: ClassVar[str] = "average"
    label: ClassVar[str] = "Average"
    needs_linear: ClassVar[bool] = True
    parameters: ClassVar[tuple] = ()

    def blend(
        self,
        accumulator: np.ndarray,
        incoming: np.ndarray,
        params: Mapping[str, Any] | None = None,
        count: int = 1,
    ) -> np.ndarray:
        n = max(int(count), 1)
        w = np.float32(1.0 / (n + 1))
        return accumulator + (incoming - accumulator) * w
