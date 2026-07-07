"""Per-image pre-blend adjustments (project brief §4.1).

Applied in this fixed order (brief §4 pipeline):

1. **Exposure trim** −3.0…+3.0 EV — linearise sRGB → multiply by ``2**EV``
   → re-encode sRGB.  Done in linear light so it behaves like a real
   exposure change, not a gamma-space gain.
2. **Contrast** −100…+100 — gamma-space pivot at 0.5:
   ``out = (in - 0.5) * k + 0.5`` with k mapping −100→0.5, 0→1.0, +100→2.0
   (``k = 2**(c/100)``, the unique smooth exponential through those points).
3. **Saturation** −100…+100 — ``out = lerp(luma, in, s)`` per pixel with
   Rec.709 luma weights (0.2126, 0.7152, 0.0722); s maps −100→0.0
   (greyscale), 0→1.0, +100→2.0 (linear: ``s = 1 + c/100``).
4. **Sharpen** — unsharp mask
   ``out = in + amount * (in - gaussian_blur(in, radius))``,
   radius 0.5–10 px (gaussian sigma), amount 0–200 %.

**Opacity** (0–100 %) is *not* an image adjustment: it is applied at the
fold step by the engine (``accumulator = lerp(accumulator, blended,
opacity)``; the first image's opacity is ignored).  It is carried in the
same per-image dict/dataclass purely for frontend convenience.

The sRGB transfer functions are the proper piecewise IEC 61966-2-1 EOTF
(linear toe + 2.4 power), not a plain gamma 2.2.  The gaussian blur is a
separable convolution implemented in pure NumPy — no UI toolkits, no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Union

import numpy as np

__all__ = [
    "Adjustments",
    "DEFAULT_ADJUSTMENTS",
    "srgb_to_linear",
    "linear_to_srgb",
    "rec709_luma",
    "apply_exposure",
    "apply_contrast",
    "apply_saturation",
    "apply_sharpen",
    "gaussian_blur",
    "apply_adjustments",
]

#: Rec.709 luma weights (brief §4.1).
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class Adjustments:
    """Per-image adjustment settings, all in UI units (brief §4.1).

    ``opacity`` is stored here for convenience but is consumed by the
    engine's fold step, never by :func:`apply_adjustments`.
    """

    exposure: float = 0.0        # EV, −3.0 … +3.0
    contrast: float = 0.0        # −100 … +100
    saturation: float = 0.0      # −100 … +100
    sharpen_radius: float = 1.0  # px (gaussian sigma), 0.5 … 10
    sharpen_amount: float = 0.0  # percent, 0 … 200
    opacity: float = 100.0       # percent, 0 … 100 (fold step, engine-owned)

    @classmethod
    def from_mapping(
        cls, mapping: Union["Adjustments", Mapping[str, Any], None]
    ) -> "Adjustments":
        """Build from a plain dict (unknown keys rejected), pass through
        an existing instance, or return defaults for ``None``."""
        if mapping is None:
            return cls()
        if isinstance(mapping, Adjustments):
            return mapping
        known = {f.name for f in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(
                f"Unknown adjustment key(s): {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**{k: float(v) for k, v in mapping.items()})

    @property
    def is_identity(self) -> bool:
        """True if the four image adjustments are all no-ops (opacity is
        irrelevant here — it acts at the fold step)."""
        return (
            self.exposure == 0.0
            and self.contrast == 0.0
            and self.saturation == 0.0
            and self.sharpen_amount == 0.0
        )


DEFAULT_ADJUSTMENTS = Adjustments()


# --------------------------------------------------------------------------
# sRGB transfer functions (IEC 61966-2-1 piecewise EOTF, brief §4.1)
# --------------------------------------------------------------------------

def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """sRGB-encoded → linear light. Piecewise linear+power EOTF.

    Negative inputs are clamped to 0; inputs above 1 follow the power-law
    extension (needed for headroom created by other adjustments).
    """
    x = np.maximum(encoded, 0.0).astype(np.float32, copy=False)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32, copy=False)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear light → sRGB-encoded. Inverse of :func:`srgb_to_linear`."""
    x = np.maximum(linear, 0.0).astype(np.float32, copy=False)
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * x ** (1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def rec709_luma(image: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an (H, W, 3) array as (H, W, 1), same dtype rules."""
    r, g, b = _LUMA_WEIGHTS
    luma = r * image[..., 0] + g * image[..., 1] + b * image[..., 2]
    return luma[..., np.newaxis]


# --------------------------------------------------------------------------
# Individual adjustments (brief §4.1 definitions)
# --------------------------------------------------------------------------

def apply_exposure(image: np.ndarray, ev: float) -> np.ndarray:
    """Exposure trim in EV, performed in linear light."""
    if ev == 0.0:
        return image
    linear = srgb_to_linear(image)
    return linear_to_srgb(linear * float(2.0 ** ev))


def apply_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    """Gamma-space contrast about pivot 0.5; k = 2**(contrast/100)."""
    if contrast == 0.0:
        return image
    k = float(2.0 ** (contrast / 100.0))
    return (image - 0.5) * k + 0.5


def apply_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
    """lerp(luma, in, s) with s = 1 + saturation/100 (−100→0, +100→2)."""
    if saturation == 0.0:
        return image
    s = 1.0 + float(saturation) / 100.0
    luma = rec709_luma(image)
    return luma + (image - luma) * s


def apply_sharpen(image: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Unsharp mask: out = in + amount × (in − gaussian_blur(in, radius))."""
    if amount == 0.0:
        return image
    radius = min(max(float(radius), 0.5), 10.0)
    blurred = gaussian_blur(image, radius)
    return image + (float(amount) / 100.0) * (image - blurred)


# --------------------------------------------------------------------------
# Separable gaussian blur — pure NumPy (brief §3 design rule 1)
# --------------------------------------------------------------------------

def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalised 1-D gaussian kernel, half-width ceil(3 sigma) (min 1)."""
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / np.float32(sigma)) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1-D convolution along ``axis`` with mirrored ('symmetric') edges."""
    half = kernel.shape[0] // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (half, half)
    padded = np.pad(image, pad, mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, kernel.shape[0], axis=axis
    )
    return np.einsum(
        "...k,k->...", windows, kernel.astype(image.dtype, copy=False)
    ).astype(image.dtype, copy=False)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur of an (H, W[, C]) array, sigma in pixels."""
    if sigma <= 0.0:
        return image
    kernel = _gaussian_kernel(sigma)
    out = _convolve_axis(image, kernel, axis=0)
    return _convolve_axis(out, kernel, axis=1)


# --------------------------------------------------------------------------
# Combined application, fixed order (brief §4 pipeline)
# --------------------------------------------------------------------------

def apply_adjustments(
    image: np.ndarray,
    adjustments: Union[Adjustments, Mapping[str, Any], None] = None,
) -> np.ndarray:
    """Apply the fixed adjustment chain exposure→contrast→saturation→sharpen.

    Identity settings return the input array *unchanged and uncopied* —
    frontends can rely on this for cache identity, and it guarantees the
    engine's lossless round-trip when no adjustments are set (brief §8).
    Non-identity results are clipped to 0–1 so the fold always receives
    in-gamut float32 data.  Opacity is deliberately ignored here (fold-step
    concern, brief §4.1).
    """
    adj = Adjustments.from_mapping(adjustments)
    if adj.is_identity:
        return image
    out = apply_exposure(image, adj.exposure)
    out = apply_contrast(out, adj.contrast)
    out = apply_saturation(out, adj.saturation)
    out = apply_sharpen(out, adj.sharpen_radius, adj.sharpen_amount)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
