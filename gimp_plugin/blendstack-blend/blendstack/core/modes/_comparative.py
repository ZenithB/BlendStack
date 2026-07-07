"""Shared maths for the Canon comparative blend modes (project brief §4.2).

Both Canon Bright and Canon Dark are one formulation differing only in
comparison direction, so the whole computation lives here.

Per pixel:

* **Comparison value** — basis ``per_channel`` (default, Canon-faithful):
  each channel compared independently, three selections per pixel, colours
  may mix (the authentic Canon artifact).  Basis ``luminance``: Rec.709 luma
  (0.2126, 0.7152, 0.0722) computed once per pixel; one selection weight
  applied to all three channels, so the winner keeps its colour intact.

* **Selection weight** (Canon Bright)::

      d = (B_cmp - A_cmp) + bias        # A = accumulator, B = incoming
      softness == 0:  w = (d > 0) ? 1 : 0        # hard max — exact Canon
      softness  > 0:  w = sigmoid(d / t)         # t linear in (0.0025, 0.25]
      out = A * (1 - w) + B * w

  Canon Dark is identical with ``d = (A_cmp - B_cmp) + bias``.

``out`` is a convex combination of A and B, so the soft version never
overshoots either source.  Bias UI range −100…+100 maps linearly to
−0.25…+0.25 normalised units, added to the incoming image's side of the
comparison (positive = incoming wins ties and near-ties more often).
Softness UI range 1–100 maps linearly onto t ∈ (0.0025, 0.25], i.e.
``t = softness / 400``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from .registry import BlendMode, ModeParameter

__all__ = ["ComparativeMode", "rec709_luma"]

#: Rec.709 luma weights (brief §4.1 / §4.2).
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: UI bias −100…+100 → −0.25…+0.25 normalised units.
_BIAS_SCALE = 0.25 / 100.0

#: UI softness 1…100 → sigmoid temperature t linear in (0.0025, 0.25].
_SOFTNESS_SCALE = 0.25 / 100.0


def rec709_luma(image: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an (H, W, 3) array, returned as (H, W, 1) float32."""
    r, g, b = _LUMA_WEIGHTS
    luma = r * image[..., 0] + g * image[..., 1] + b * image[..., 2]
    return luma[..., np.newaxis]


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically safe logistic sigmoid (clips the exponent, keeps dtype)."""
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class ComparativeMode(BlendMode):
    """Common implementation of Canon Bright / Canon Dark (brief §4.2)."""

    #: +1.0 selects the *brighter* comparison value (Canon Bright);
    #: -1.0 selects the *darker* one (Canon Dark).
    _direction: ClassVar[float]

    needs_linear: ClassVar[bool] = False  # max/min are monotone-invariant (brief §1)

    parameters: ClassVar[tuple[ModeParameter, ...]] = (
        ModeParameter("softness", default=0.0, min_value=0.0, max_value=100.0,
                      label="Softness"),
        ModeParameter("bias", default=0.0, min_value=-100.0, max_value=100.0,
                      label="Bias"),
        ModeParameter("basis", default="per_channel",
                      choices=("per_channel", "luminance"),
                      label="Comparison basis"),
    )

    def blend(
        self,
        accumulator: np.ndarray,
        incoming: np.ndarray,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        p = self.resolve_params(params)
        softness = float(p["softness"])
        bias = float(p["bias"]) * _BIAS_SCALE
        basis = str(p["basis"])
        if basis not in ("per_channel", "luminance"):
            raise ValueError(f"Unknown comparison basis '{basis}'")

        a, b = accumulator, incoming

        # Fast path: all defaults, per-channel — mathematically identical to
        # the general formulation below and bit-exact against np.maximum /
        # np.minimum (brief §8 acceptance criterion, order-independent fold).
        if softness <= 0.0 and bias == 0.0 and basis == "per_channel":
            return np.maximum(a, b) if self._direction > 0 else np.minimum(a, b)

        if basis == "luminance":
            a_cmp: np.ndarray = rec709_luma(a)
            b_cmp: np.ndarray = rec709_luma(b)
        else:
            a_cmp, b_cmp = a, b

        # Signed comparison; bias is added to the incoming image's side.
        if self._direction > 0:
            d = (b_cmp - a_cmp) + bias
        else:
            d = (a_cmp - b_cmp) + bias

        if softness <= 0.0:
            # Hard select: w = (d > 0) ? 1 : 0 — ties keep the accumulator.
            return np.where(d > 0.0, b, a)

        t = softness * _SOFTNESS_SCALE  # t = softness / 400, in (0.0025, 0.25]
        w = _stable_sigmoid(d / t)
        # Convex combination — never overshoots either source.
        return a + (b - a) * w
