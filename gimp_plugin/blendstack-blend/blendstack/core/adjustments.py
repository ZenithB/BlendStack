"""Per-image pre-blend adjustments (project brief §4.1).

Applied in the order given by ``Adjustments.order`` (default below; brief §4
pipeline plus the curves and denoise stages added in v1.1):

1. **Exposure trim** −3.0…+3.0 EV — linearise sRGB → multiply by ``2**EV``
   → re-encode sRGB.  Done in linear light so it behaves like a real
   exposure change, not a gamma-space gain.
2. **Contrast** −100…+100 — gamma-space pivot at 0.5:
   ``out = (in - 0.5) * k + 0.5`` with k mapping −100→0.5, 0→1.0, +100→2.0
   (``k = 2**(c/100)``, the unique smooth exponential through those points).
3. **Curves** — master + per-channel R/G/B tone curves defined by control
   points, monotone cubic (PCHIP) interpolated into a 4096-entry LUT.
   The master curve runs first (all channels), then the R/G/B curves.
4. **Saturation** −100…+100 — ``out = lerp(luma, in, s)`` per pixel with
   Rec.709 luma weights (0.2126, 0.7152, 0.0722); s maps −100→0.0
   (greyscale), 0→1.0, +100→2.0 (linear: ``s = 1 + c/100``).
5. **Denoise** 0…100 — edge-preserving guided filter (He et al.), implemented
   with O(N) cumulative-sum box filters.
6. **Sharpen** — unsharp mask
   ``out = in + amount * (in - gaussian_blur(in, radius))``,
   radius 0.5–20 px (gaussian sigma), amount 0–200 %.

``Adjustments.order`` lets the user permute these per-image stages; the
default is exactly the order above (identical to the pre-v1.1 chain for
images that only use the old fields).  Each stage is a no-op at its
identity setting.

**Opacity** (0–100 %) and the layer placement offsets ``move_x`` / ``move_y``
are *not* image adjustments: opacity is applied at the fold step by the
engine (``accumulator = lerp(accumulator, blended, opacity)``; the first
image's opacity is ignored) and the placement offset is consumed by
geometry/engine.  They are carried in the same per-image dict/dataclass
purely for frontend convenience.

The sRGB transfer functions are the proper piecewise IEC 61966-2-1 EOTF
(linear toe + 2.4 power), not a plain gamma 2.2.  The gaussian blur is a
separable convolution implemented in pure NumPy — no UI toolkits, no scipy
(an exact kernel for small sigma, a 3-pass box approximation above
``_FAST_BLUR_SIGMA``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Union

import numpy as np

__all__ = [
    "Adjustments",
    "DEFAULT_ADJUSTMENTS",
    "STAGES",
    "DEFAULT_ORDER",
    "IDENTITY_CURVE",
    "srgb_to_linear",
    "linear_to_srgb",
    "rec709_luma",
    "apply_exposure",
    "apply_contrast",
    "apply_saturation",
    "apply_sharpen",
    "apply_curves",
    "apply_denoise",
    "curve_lut",
    "gaussian_blur",
    "apply_adjustments",
]

#: Rec.709 luma weights (brief §4.1).
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: Per-image stage names, in default processing order.
STAGES = ("exposure", "contrast", "curves", "saturation", "denoise", "sharpen")
DEFAULT_ORDER = STAGES

#: Sharpen radius range (gaussian sigma, px).
SHARPEN_RADIUS_MIN = 0.5
SHARPEN_RADIUS_MAX = 20.0

CurvePoints = tuple  # tuple[tuple[float, float], ...]

#: The identity tone curve: two control points on the diagonal.
IDENTITY_CURVE: CurvePoints = ((0.0, 0.0), (1.0, 1.0))

_CURVE_FIELDS = ("curve_master", "curve_red", "curve_green", "curve_blue")


def _normalise_curve(name: str, points: Any) -> CurvePoints:
    """Validate control points and return them as a sorted tuple of
    ``(x, y)`` float tuples.  Raises ``ValueError`` on bad input."""
    try:
        pts = []
        for p in points:
            if len(p) != 2:
                raise TypeError
            pts.append((float(p[0]), float(p[1])))
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}: curve must be a sequence of (x, y) number pairs"
        ) from None
    if len(pts) < 2:
        raise ValueError(f"{name}: a curve needs at least 2 control points")
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)
                and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"{name}: control point ({x}, {y}) outside the 0..1 range"
            )
    pts.sort(key=lambda p: p[0])
    for (x0, _), (x1, _) in zip(pts, pts[1:]):
        if not x1 > x0:
            raise ValueError(
                f"{name}: control point x values must be strictly increasing "
                f"(duplicate x = {x1})"
            )
    return tuple(pts)


def _normalise_order(order: Any) -> tuple:
    """Drop unknown/duplicate stage names and append missing stages in
    default order, so any permutation or truncated list is valid."""
    if isinstance(order, str):
        order = (order,)
    seen: list = []
    for name in order or ():
        if isinstance(name, str) and name in STAGES and name not in seen:
            seen.append(name)
    seen.extend(s for s in DEFAULT_ORDER if s not in seen)
    return tuple(seen)


def _is_identity_curve(points: CurvePoints) -> bool:
    """True for a curve that is exactly y = x over the whole 0..1 range."""
    return (
        points[0][0] == 0.0
        and points[-1][0] == 1.0
        and all(x == y for x, y in points)
    )


@dataclass(frozen=True)
class Adjustments:
    """Per-image adjustment settings, all in UI units (brief §4.1).

    ``opacity`` is stored here for convenience but is consumed by the
    engine's fold step, never by :func:`apply_adjustments`.  Likewise
    ``move_x`` / ``move_y`` are layer *placement* (fractions of the canvas
    width / height, +x right, +y down) consumed by geometry/engine.
    """

    exposure: float = 0.0        # EV, −3.0 … +3.0
    contrast: float = 0.0        # −100 … +100
    saturation: float = 0.0      # −100 … +100
    sharpen_radius: float = 1.0  # px (gaussian sigma), 0.5 … 20
    sharpen_amount: float = 0.0  # percent, 0 … 200
    opacity: float = 100.0       # percent, 0 … 100 (fold step, engine-owned)
    denoise: float = 0.0         # 0 … 100 noise-removal strength
    curve_master: CurvePoints = IDENTITY_CURVE   # (x, y) control points, 0..1
    curve_red: CurvePoints = IDENTITY_CURVE
    curve_green: CurvePoints = IDENTITY_CURVE
    curve_blue: CurvePoints = IDENTITY_CURVE
    order: tuple = DEFAULT_ORDER  # processing order of the per-image stages
    move_x: float = 0.0          # placement, fraction of canvas width (+right)
    move_y: float = 0.0          # placement, fraction of canvas height (+down)
    mute: bool = False           # excluded from the blend (still sizes the canvas);
                                 # like opacity/placement it is NOT an image adjustment

    def __post_init__(self) -> None:
        # Normalise (and validate) curves and order so JSON lists / any
        # permutation are accepted and the instance stays hashable.
        for name in _CURVE_FIELDS:
            object.__setattr__(
                self, name, _normalise_curve(name, getattr(self, name))
            )
        object.__setattr__(self, "order", _normalise_order(self.order))

    @classmethod
    def from_mapping(
        cls, mapping: Union["Adjustments", Mapping[str, Any], None]
    ) -> "Adjustments":
        """Build from a plain dict (unknown keys rejected), pass through
        an existing instance, or return defaults for ``None``.

        Missing keys take defaults, so presets saved before v1.1 (six
        keys) still load.  JSON lists are accepted for curves and
        ``order``; curves are validated (``ValueError``) and ``order`` is
        normalised (unknown/duplicate names dropped, missing appended)."""
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
        kwargs: dict = {}
        for k, v in mapping.items():
            if k in _CURVE_FIELDS or k == "order":
                kwargs[k] = v  # normalised/validated by __post_init__
            elif k == "mute":
                kwargs[k] = bool(v)
            else:
                kwargs[k] = float(v)
        return cls(**kwargs)

    @property
    def is_identity(self) -> bool:
        """True if every per-image adjustment is a no-op (opacity and the
        move_x/move_y placement are irrelevant here — they act at the fold /
        placement step)."""
        return (
            self.exposure == 0.0
            and self.contrast == 0.0
            and self.saturation == 0.0
            and self.sharpen_amount == 0.0
            and self.denoise == 0.0
            and all(_is_identity_curve(getattr(self, n)) for n in _CURVE_FIELDS)
        )

    def adjust_key(self) -> tuple:
        """Hashable key of everything that affects the per-image adjusted
        pixels.  Excludes ``opacity``, ``mute``, ``move_x`` and ``move_y`` (fold /
        placement concerns) so it is a valid cache key for adjusted images."""
        return (
            self.exposure, self.contrast, self.saturation, self.denoise,
            self.sharpen_radius, self.sharpen_amount,
            self.curve_master, self.curve_red, self.curve_green,
            self.curve_blue, self.order,
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
    """Unsharp mask: out = in + amount × (in − gaussian_blur(in, radius)).

    ``radius`` (gaussian sigma, px) is clamped to 0.5 … 20."""
    if amount == 0.0:
        return image
    radius = min(max(float(radius), SHARPEN_RADIUS_MIN), SHARPEN_RADIUS_MAX)
    blurred = gaussian_blur(image, radius)
    return image + (float(amount) / 100.0) * (image - blurred)


# --------------------------------------------------------------------------
# Tone curves (monotone cubic / PCHIP LUTs)
# --------------------------------------------------------------------------

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Butland (PCHIP) node slopes: weighted harmonic mean of the
    neighbouring secants, zero where the secants change sign (so the
    interpolant is monotone between data extrema and never overshoots),
    with shape-preserving three-point end slopes."""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n, dtype=np.float64)
    if n == 2:
        d[:] = delta[0]
        return d
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    prod = delta[:-1] * delta[1:]
    ok = prod > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    d[1:-1] = np.where(ok, hm, 0.0)

    def end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > 3.0 * abs(d0):
            return 3.0 * d0
        return float(s)

    d[0] = end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def curve_lut(points, size: int = 4096) -> np.ndarray:
    """Monotone cubic (PCHIP / Fritsch–Carlson family) interpolation through
    the control ``points`` ``((x, y), ...)``, sampled at ``size`` evenly
    spaced x in 0..1.  Flat (clamped to the end y) outside the first/last
    point's x, and never overshoots the neighbouring control-point y values.
    Returns float32 of shape ``(size,)``.  Pure NumPy."""
    pts = _normalise_curve("points", points)
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    d = _pchip_slopes(x, y)
    g = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    idx = np.clip(np.searchsorted(x, g, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = np.clip((g - x[idx]) / h, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    out = (
        (2 * t3 - 3 * t2 + 1) * y[idx]
        + (t3 - 2 * t2 + t) * h * d[idx]
        + (-2 * t3 + 3 * t2) * y[idx + 1]
        + (t3 - t2) * h * d[idx + 1]
    )
    out = np.where(g <= x[0], y[0], out)
    out = np.where(g >= x[-1], y[-1], out)
    # Guard against last-ulp float excursions beyond the control-point range.
    return np.clip(out, y.min(), y.max()).astype(np.float32)


_LUT_SIZE = 4096
_LOOKUP_CHUNK = 1 << 18  # elements per lookup block (cache-friendly)


def _lut_lookup(plane: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Linear-interpolated LUT lookup of ``plane`` (any shape; inputs
    clipped to 0..1) into a float32 array of the same shape, in blocks."""
    n = lut.shape[0]
    lut = np.ascontiguousarray(lut, dtype=np.float32)
    dlut = np.diff(lut, append=lut[-1])  # dlut[n-1] = 0: top entry is exact
    flat = plane.reshape(-1)
    out = np.empty(flat.shape, dtype=np.float32)
    scale = np.float32(n - 1)
    for s in range(0, flat.size, _LOOKUP_CHUNK):
        pos = np.clip(flat[s : s + _LOOKUP_CHUNK], 0.0, 1.0).astype(np.float32)
        pos *= scale
        idx = pos.astype(np.intp)
        pos -= idx
        pos *= dlut.take(idx)
        pos += lut.take(idx)
        out[s : s + _LOOKUP_CHUNK] = pos
    return out.reshape(plane.shape)


def apply_curves(image: np.ndarray, master, red, green, blue) -> np.ndarray:
    """Apply tone curves to an (H, W, 3) image: the MASTER curve to all
    channels first, then the per-channel R/G/B curves.  Inputs are clipped
    to 0..1 for the lookup (linear interpolation in a 4096-entry LUT).
    Returns the input object untouched when all four curves are identity.

    Master-then-channel is evaluated as one lookup per channel through the
    composed LUT ``lut_channel(lut_master(x))`` (composition is done on the
    LUT grid, so the cost is one pass over the pixels per channel)."""
    curves = [_normalise_curve("curve", c) for c in (master, red, green, blue)]
    if all(_is_identity_curve(c) for c in curves):
        return image
    grid = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)
    master_lut = (
        None if _is_identity_curve(curves[0]) else curve_lut(curves[0], _LUT_SIZE)
    )
    out = np.empty(image.shape, dtype=np.float32)
    for ch, c in enumerate(curves[1:]):
        lut = None if _is_identity_curve(c) else curve_lut(c, _LUT_SIZE)
        if master_lut is not None and lut is not None:
            lut = np.interp(master_lut, grid, lut).astype(np.float32)
        elif lut is None:
            lut = master_lut
        if lut is None:  # neither master nor this channel has a curve
            out[..., ch] = np.clip(image[..., ch], 0.0, 1.0)
        else:
            out[..., ch] = _lut_lookup(image[..., ch], lut)
    return out


# --------------------------------------------------------------------------
# Denoise — guided filter (He, Sun & Tang), O(N) cumulative-sum box filters
# --------------------------------------------------------------------------

#: Float64 elements per block when running cumulative sums (bounds memory).
_BLOCK_ELEMS = 1 << 22


def _box_mean_axis(x: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Mean over a (2r+1) window along ``axis`` with the window clipped at
    the borders (so tiny arrays work and edges are unbiased).  float64
    cumulative sum: O(N), independent of radius."""
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    c = np.cumsum(x, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + c.shape[1:], dtype=np.float64), c], axis=0)
    i = np.arange(n)
    hi = np.minimum(i + radius + 1, n)
    lo = np.maximum(i - radius, 0)
    cnt = (hi - lo).astype(np.float64).reshape((n,) + (1,) * (x.ndim - 1))
    out = (c[hi] - c[lo]) / cnt
    return np.moveaxis(out, 0, axis)


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    """2-D clipped box mean of an (H, W, C) array (separable)."""
    return _box_mean_axis(_box_mean_axis(x, radius, 0), radius, 1)


def _guided_self(img: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided guided filter, independently per channel.  ``img`` is
    (h, w, C) float64.  a = var/(var+eps), b = (1-a)·mean,
    out = mean(a)·I + mean(b)."""
    mean_i = _box_mean(img, radius)
    var = np.maximum(_box_mean(img * img, radius) - mean_i * mean_i, 0.0)
    a = var / (var + eps)
    b = mean_i * (1.0 - a)
    return _box_mean(a, radius) * img + _box_mean(b, radius)


#: Calibration multiplier on the nominal eps (0.02 + 0.2 t)^2, t = strength/100.
#: Chosen so a sigma-0.03 noisy 0.2→0.8 step loses ≥50 % of its flat-region
#: noise std at strength 50 (measured ≈80 %) with a 10–90 % edge width under
#: 1 px at every strength (see tests/test_curves_denoise.py).
_EPS_SCALE = 0.3


def _denoise_params(strength: float) -> tuple[int, float]:
    """Map strength 0..100 → (box radius px, eps)."""
    t = min(max(float(strength), 0.0), 100.0) / 100.0
    radius = max(1, int(math.floor(1.0 + 3.0 * t + 0.5)))
    eps = _EPS_SCALE * (0.02 + 0.2 * t) ** 2
    return radius, eps


def apply_denoise(image: np.ndarray, strength: float,
                  _band_rows: int | None = None) -> np.ndarray:
    """Edge-preserving noise removal, strength 0..100 (0 returns the input
    object untouched).

    A guided filter (He et al.), *self-guided per channel*: each channel is
    its own guide, so every channel keeps its own edges, there is no
    cross-channel colour bleeding and no luma assumption is baked in.  Box
    radius grows 1 → 4 px and eps grows with strength.  The image is
    processed in horizontal bands with a 2·radius halo, which gives results
    identical to whole-image processing while bounding float64 memory on
    24 MP images."""
    strength = float(strength)
    if strength <= 0.0:
        return image
    radius, eps = _denoise_params(strength)
    arr = np.asarray(image)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[..., np.newaxis]
    h, w = arr.shape[:2]
    c = arr.shape[2]
    out = np.empty(arr.shape, dtype=np.float32)
    band = _band_rows or max(16, _BLOCK_ELEMS // max(1, w * c))
    halo = 2 * radius
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        s0, s1 = max(0, y0 - halo), min(h, y1 + halo)
        sub = arr[s0:s1].astype(np.float64)
        res = _guided_self(sub, radius, eps)
        out[y0:y1] = res[y0 - s0 : y0 - s0 + (y1 - y0)]
    return out[..., 0] if squeeze else out


# --------------------------------------------------------------------------
# Gaussian blur — pure NumPy (brief §3 design rule 1)
# --------------------------------------------------------------------------

#: Above this sigma the O(N) 3-pass box approximation replaces the exact
#: kernel (whose cost grows linearly with sigma).
_FAST_BLUR_SIGMA = 3.0


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


def _box_radii(sigma: float, n: int = 3) -> list[int]:
    """Radii of ``n`` odd-width boxes whose cascade approximates a gaussian
    of ``sigma`` (the standard "gaussian by boxes" construction: two widths
    wl (odd, ≤ ideal) and wl+2, split so the total variance matches)."""
    w_ideal = math.sqrt(12.0 * sigma * sigma / n + 1.0)
    wl = int(math.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m_ideal = (12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / (
        -4.0 * wl - 4.0
    )
    m = int(round(m_ideal))
    widths = [wl if i < m else wu for i in range(n)]
    return [(wd - 1) // 2 for wd in widths]


def _box_blur_axis(image: np.ndarray, radii: list[int], axis: int) -> np.ndarray:
    """Cascade of box blurs (radii) along one spatial axis with symmetric
    (mirrored) edge handling, O(N) per pass via float64 cumulative sums,
    processed in blocks of the other spatial axis to bound memory."""
    total = sum(radii)
    other = 1 - axis
    n_other = image.shape[other]
    per_line = (image.shape[axis] + 2 * total) * (
        image.size // max(1, image.shape[axis] * n_other)
    )
    chunk = max(1, _BLOCK_ELEMS // max(1, per_line))
    out = np.empty_like(image)
    for s in range(0, n_other, chunk):
        sl = [slice(None)] * image.ndim
        sl[other] = slice(s, s + chunk)
        block = image[tuple(sl)]
        pad = [(0, 0)] * image.ndim
        pad[axis] = (total, total)
        cur = np.moveaxis(np.pad(block, pad, mode="symmetric"), axis, 0)
        cur = cur.astype(np.float64)
        for r in radii:
            wd = 2 * r + 1
            c = np.cumsum(cur, axis=0)
            res = c[wd - 1 :].copy()
            res[1:] -= c[: res.shape[0] - 1]
            res /= wd
            cur = res
        out[tuple(sl)] = np.moveaxis(cur, 0, axis).astype(image.dtype, copy=False)
    return out


def _gaussian_blur_fast(image: np.ndarray, sigma: float) -> np.ndarray:
    radii = _box_radii(sigma, 3)
    out = _box_blur_axis(image, radii, axis=0)
    return _box_blur_axis(out, radii, axis=1)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur of an (H, W[, C]) array, sigma in pixels.

    Exact kernel for ``sigma <= 3``; above that a 3-pass box-blur
    approximation (O(N), independent of sigma) is used."""
    if sigma <= 0.0:
        return image
    if sigma > _FAST_BLUR_SIGMA:
        return _gaussian_blur_fast(image, sigma)
    kernel = _gaussian_kernel(sigma)
    out = _convolve_axis(image, kernel, axis=0)
    return _convolve_axis(out, kernel, axis=1)


# --------------------------------------------------------------------------
# Combined application, user-defined stage order (brief §4 pipeline)
# --------------------------------------------------------------------------

def apply_adjustments(
    image: np.ndarray,
    adjustments: Union[Adjustments, Mapping[str, Any], None] = None,
) -> np.ndarray:
    """Apply the per-image stages in ``adjustments.order`` (default
    exposure→contrast→curves→saturation→denoise→sharpen); each stage is a
    no-op at its identity setting.

    Identity settings return the input array *unchanged and uncopied* —
    frontends can rely on this for cache identity, and it guarantees the
    engine's lossless round-trip when no adjustments are set (brief §8).
    Non-identity results are clipped to 0–1 so the fold always receives
    in-gamut float32 data.  Opacity and placement (``move_x``/``move_y``) are
    deliberately ignored here (fold/placement concerns, brief §4.1).
    """
    adj = Adjustments.from_mapping(adjustments)
    if adj.is_identity:
        return image
    stages = {
        "exposure": lambda im: apply_exposure(im, adj.exposure),
        "contrast": lambda im: apply_contrast(im, adj.contrast),
        "curves": lambda im: apply_curves(
            im, adj.curve_master, adj.curve_red, adj.curve_green, adj.curve_blue
        ),
        "saturation": lambda im: apply_saturation(im, adj.saturation),
        "denoise": lambda im: apply_denoise(im, adj.denoise),
        "sharpen": lambda im: apply_sharpen(
            im, adj.sharpen_radius, adj.sharpen_amount
        ),
    }
    out = image
    for name in adj.order:
        out = stages[name](out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
