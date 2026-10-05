"""Pipeline orchestration (project brief §4).

For each render (preview or export)::

    for each image i in user-defined order:
        load -> float32 RGB 0..1        (RAW via LibRaw defaults; GIF frame 0;
                                         alpha flattened against black)
        geometry: cover-scale (no crop) to target dims (smallest by area)
        adjustments, in Adjustments.order (default: exposure -> contrast ->
            curves -> saturation -> denoise -> sharpen) on the scaled image
        placement: centre + shift by (move_x, move_y); the uncovered gap is
            TRANSPARENT (coverage mask), never black
    accumulator = image[0]              (opacity ignored for the first image)
    for each subsequent image i:
        blended = mode.blend(accumulator, image[i], params)
        accumulator = lerp(accumulator, blended, opacity[i])
    clip accumulator to 0..1
    optional crop (fractions of the canvas)
    encode to output format / bit depth

Public API for frontends (GUI and GIMP plugin):

* :func:`blend_arrays` — already-loaded float32 arrays in, composite out.
* :func:`blend_files`  — file paths in, saved output file out (streams one
  image at a time, so 20 × 24 MP stays within a small memory budget).
* :func:`conform_images` / :func:`adjust_image` — the geometry and
  adjustment steps exposed separately so a frontend can cache per-image
  intermediate results and re-run only the fold on slider drags (brief §5).
* :class:`BlendFold` — incremental fold, one image at a time (coverage
  aware: ``push(image, opacity, mask)``).
* :func:`apply_crop` / :func:`crop_box_px` — crop the composite by fractions.

Linear-light scaffolding (brief §3 design rule 2): if the selected mode
declares ``needs_linear = True``, the fold inputs are linearised with the
sRGB EOTF before folding and the accumulator is re-encoded afterwards.
Both v1 modes declare False (max/min are monotone-invariant, brief §1).

All opacity and mode parameters are in UI units throughout (opacity
0–100 %, softness 0–100, bias −100…+100).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import adjustments as adj_mod
from . import geometry
from . import io as bs_io  # io.py shadows stdlib io inside the package
from .adjustments import Adjustments
from .modes import get_mode, mode_names
from .modes.registry import BlendMode

__all__ = [
    "MIN_IMAGES",
    "MAX_IMAGES",
    "BlendFold",
    "conform_images",
    "adjust_image",
    "fold_images",
    "apply_crop",
    "crop_box_px",
    "blend_arrays",
    "blend_files",
    "get_mode",
    "mode_names",
]

#: Blend size limits (brief §2: 2–20 input images per blend).
MIN_IMAGES = 2
MAX_IMAGES = 20

PathLike = Union[str, Path]
AdjustmentsLike = Union[Adjustments, Mapping[str, Any], None]


# --------------------------------------------------------------------------
# Individually exposed pipeline steps (for frontend preview caching)
# --------------------------------------------------------------------------

def conform_images(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Geometry step (brief §4.3): conform all images to the smallest
    input's dimensions by area (cover-scale + centre-crop, Lanczos).
    Identical-size inputs pass through untouched (no copy)."""
    return geometry.conform_stack(images)


def adjust_image(image: np.ndarray, adjustments: AdjustmentsLike = None) -> np.ndarray:
    """Adjustment step (brief §4.1) for one image: exposure → contrast →
    saturation → sharpen, in that fixed order.  Identity settings return
    the input array unchanged (no copy) — safe to use as a cache key.
    Opacity in ``adjustments`` is ignored here; it acts at the fold step."""
    return adj_mod.apply_adjustments(image, adjustments)


class BlendFold:
    """Incremental left fold of images into a composite (brief §4).

    Push geometry-conformed, adjusted float32 RGB (H, W, 3) images one at a
    time, then call :meth:`result`.  Handles the linear-light scaffolding
    for modes with ``needs_linear = True`` and enforces the 20-image cap.

    The first pushed image becomes the accumulator; its opacity is ignored.
    Each further image is folded as
    ``accumulator = lerp(accumulator, mode.blend(accumulator, image), opacity)``.

    **Coverage.**  A layer moved off the canvas leaves an uncovered gap
    (see :func:`blendstack.core.geometry.place`).  Push it with its
    ``mask``: uncovered area is *transparent* — it neither contributes to
    the blend nor acts as black.  Per pixel: both covered → normal blend +
    opacity lerp; only the incoming covered → incoming pixel taken as is;
    only the accumulator covered → unchanged; neither → stays uncovered
    (written as black by :meth:`result`).  While no mask has ever been
    pushed the original fast path runs unchanged (and ``count`` passed to
    ``mode.blend`` is an int); once coverage tracking is active ``count`` is
    a float32 ``(H, W, 1)`` array of layers folded into each pixel.
    """

    def __init__(self, mode: str = "canon_bright",
                 params: Mapping[str, Any] | None = None) -> None:
        self._mode: BlendMode = get_mode(mode)
        # Lenient: frontends pass one global param dict (softness/bias/basis)
        # across all modes; keep only the keys THIS mode declares so a
        # parameter-free mode (Multiply, Screen, …) does not choke on them.
        self._params: dict[str, Any] = self._mode.pick_params(params)
        self._accumulator: Optional[np.ndarray] = None
        self._count = 0
        # Coverage tracking (None => every pixel covered; fast path).
        self._cov: Optional[np.ndarray] = None       # bool (H, W, 1)
        self._counts: Optional[np.ndarray] = None    # float32 (H, W, 1)

    @property
    def count(self) -> int:
        """Number of images pushed so far."""
        return self._count

    @property
    def coverage(self) -> Optional[np.ndarray]:
        """Boolean (H, W, 1) accumulator coverage, or ``None`` when every
        pixel is covered."""
        if self._cov is None or self._cov.all():
            return None
        return self._cov

    def push(self, image: np.ndarray, opacity: float = 100.0,
             mask: Optional[np.ndarray] = None) -> None:
        """Fold one image in. ``opacity`` is 0–100 (ignored for the first
        image).  ``mask`` is the (H, W, 1) float32 coverage from
        :func:`blendstack.core.geometry.place` (``None`` = fully covered).
        Raises ``ValueError`` past the 20-image cap."""
        if self._count >= MAX_IMAGES:
            raise ValueError(f"Blend cap exceeded: at most {MAX_IMAGES} images")
        image = _as_float_rgb(image)
        inc_cov = _coverage_bool(mask, image.shape)
        if self._mode.needs_linear:
            image = adj_mod.srgb_to_linear(image)
        if self._accumulator is None:
            if inc_cov is None:
                self._accumulator = image  # first image: opacity ignored (§4)
            else:
                self._accumulator = np.where(inc_cov, image, np.float32(0.0))
                self._cov = inc_cov
                self._counts = inc_cov.astype(np.float32)
        else:
            if self._accumulator.shape != image.shape:
                raise ValueError(
                    f"Image shape {image.shape} does not match accumulator "
                    f"{self._accumulator.shape}; run the geometry step first"
                )
            if self._cov is None and inc_cov is None:
                self._push_covered(image, opacity)
            else:
                self._push_with_coverage(image, opacity, inc_cov)
        self._count += 1

    def _lerp(self, base: np.ndarray, blended: np.ndarray, opacity: float):
        k = float(opacity) / 100.0
        if k >= 1.0:
            return blended
        return base + (blended - base) * np.float32(k)

    def _push_covered(self, image: np.ndarray, opacity: float) -> None:
        """Original fast path: everything covered, int ``count``."""
        # count = images already folded into the accumulator, so the
        # incoming image is number count+1 (needed by Average).
        blended = self._mode.blend(
            self._accumulator, image, self._params, count=self._count
        )
        self._accumulator = self._lerp(self._accumulator, blended, opacity)

    def _push_with_coverage(self, image: np.ndarray, opacity: float,
                            inc_cov: Optional[np.ndarray]) -> None:
        acc = self._accumulator
        if self._cov is None:  # entering coverage mode: acc was fully covered
            self._cov = np.ones(acc.shape[:2] + (1,), dtype=bool)
            self._counts = np.full(acc.shape[:2] + (1,), self._count,
                                   dtype=np.float32)
        acc_cov = self._cov
        if inc_cov is None:
            inc_cov = np.ones_like(acc_cov)
        both = acc_cov & inc_cov
        only_inc = inc_cov & ~acc_cov
        if both.any():
            blended = self._mode.blend(acc, image, self._params,
                                       count=self._counts)
            merged = self._lerp(acc, blended, opacity)
            new_acc = np.where(both, merged, acc)
        else:
            new_acc = acc
        if only_inc.any():
            new_acc = np.where(only_inc, image, new_acc)
        self._accumulator = new_acc.astype(np.float32, copy=False)
        self._counts = self._counts + inc_cov
        self._cov = acc_cov | inc_cov

    def result(self) -> np.ndarray:
        """Re-encode (if the mode ran in linear light), clip to 0–1 and
        return the composite as a new float32 array.  Pixels no layer
        covers are written as 0 (black)."""
        if self._accumulator is None:
            raise ValueError("No images have been pushed")
        out = self._accumulator
        if self._mode.needs_linear:
            out = adj_mod.linear_to_srgb(out)
        # np.clip allocates a fresh array, so the accumulator stays untouched.
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        if self._cov is not None and not self._cov.all():
            out = np.where(self._cov, out, np.float32(0.0))
        return out


def fold_images(
    images: Sequence[np.ndarray],
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
    opacities: Sequence[float] | None = None,
    masks: Sequence[Optional[np.ndarray]] | None = None,
) -> np.ndarray:
    """Fold step only: blend pre-conformed, pre-adjusted images.

    ``opacities`` are 0–100 per image (first entry ignored, default all
    100).  ``masks`` are optional per-image (H, W, 1) coverage masks from
    :func:`blendstack.core.geometry.place` (``None`` entries = fully
    covered).  Returns the clipped float32 composite."""
    if opacities is not None and len(opacities) != len(images):
        raise ValueError("opacities must have one entry per image")
    if masks is not None and len(masks) != len(images):
        raise ValueError("masks must have one entry per image")
    fold = BlendFold(mode, params)
    for i, image in enumerate(images):
        fold.push(
            image,
            100.0 if opacities is None else opacities[i],
            None if masks is None else masks[i],
        )
    return fold.result()


# --------------------------------------------------------------------------
# Crop (fractions of the canvas)
# --------------------------------------------------------------------------

CropFractions = Tuple[float, float, float, float]


def _validate_crop(crop: Sequence[float]) -> CropFractions:
    try:
        x0, y0, x1, y1 = (float(v) for v in crop)
    except (TypeError, ValueError):
        raise ValueError(
            "crop must be four numbers (x0, y0, x1, y1) as 0..1 fractions"
        ) from None
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise ValueError(f"crop fractions must lie within 0..1, got {tuple(crop)}")
    if not (x0 < x1 and y0 < y1):
        raise ValueError(
            f"crop must satisfy x0 < x1 and y0 < y1, got {tuple(crop)}"
        )
    return x0, y0, x1, y1


def _round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


def crop_box_px(crop: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Integer pixel box ``(x0, y0, x1, y1)`` (x1/y1 exclusive) for crop
    fractions ``(x0, y0, x1, y1)`` of a ``width`` x ``height`` canvas:
    rounded to nearest, clamped to the canvas and at least 1x1.
    ``ValueError`` if the fractions are invalid (x0 < x1, y0 < y1, 0..1)."""
    fx0, fy0, fx1, fy1 = _validate_crop(crop)
    x0 = min(max(_round_half_up(fx0 * width), 0), width - 1)
    y0 = min(max(_round_half_up(fy0 * height), 0), height - 1)
    x1 = min(max(_round_half_up(fx1 * width), x0 + 1), width)
    y1 = min(max(_round_half_up(fy1 * height), y0 + 1), height)
    return x0, y0, x1, y1


def apply_crop(composite: np.ndarray,
               crop: Optional[Sequence[float]]) -> np.ndarray:
    """Crop ``composite`` (H, W, ...) by fractions ``(x0, y0, x1, y1)`` of
    the canvas.  ``None`` returns the input object; otherwise a C-contiguous
    *copy* of the sub-array (see :func:`crop_box_px` for the pixel box)."""
    if crop is None:
        return composite
    height, width = composite.shape[:2]
    x0, y0, x1, y1 = crop_box_px(crop, width, height)
    return np.array(composite[y0:y1, x0:x1], order="C", copy=True)


# --------------------------------------------------------------------------
# High-level API
# --------------------------------------------------------------------------

def _prepare_layer(
    scaled: np.ndarray, adj: Adjustments, target: Tuple[int, int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Adjust the cover-scaled layer, then place it on the canvas (adjust
    BEFORE place so uncovered gaps are never adjusted)."""
    adjusted = adjust_image(scaled, adj)
    dx, dy = geometry.offset_px(adj.move_x, adj.move_y, target)
    return geometry.place(adjusted, target, dx, dy)


def blend_arrays(
    images: Sequence[np.ndarray],
    adjustments: Sequence[AdjustmentsLike] | None = None,
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
    crop: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Full pipeline on already-loaded images: cover-scale → adjustments →
    placement → fold → clip → crop (brief §4).  Returns the float32
    composite (H, W, 3), 0–1.

    ``images``      — 2–20 float32 RGB arrays, (H, W, 3), values 0–1.
    ``adjustments`` — optional, one per image: :class:`Adjustments` or a
                      dict with any :class:`Adjustments` key (UI units, incl.
                      ``move_x``/``move_y`` placement fractions); ``None``
                      entries mean defaults.
    ``mode``        — registered mode name (``"canon_bright"``/``"canon_dark"``…).
    ``params``      — mode parameters, e.g. ``{"softness": 20, "bias": -10,
                      "basis": "luminance"}``; omitted keys take defaults.
    ``crop``        — optional ``(x0, y0, x1, y1)`` fractions (0..1) of the
                      canvas applied to the composite.
    """
    _check_count(len(images))
    adjs = _resolve_adjustment_list(adjustments, len(images))
    if crop is not None:
        _validate_crop(crop)
    arrays = [_as_float_rgb(im) for im in images]
    target = geometry.target_dimensions([(a.shape[1], a.shape[0]) for a in arrays])
    fold = BlendFold(mode, params)
    for array, adj in zip(arrays, adjs):
        placed, mask = _prepare_layer(geometry.cover_scale(array, target), adj, target)
        fold.push(placed, opacity=adj.opacity, mask=mask)
    return apply_crop(fold.result(), crop)


def blend_files(
    paths: Sequence[PathLike],
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
    adjustments: Sequence[AdjustmentsLike] | None = None,
    out_path: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    out_format: Optional[str] = None,
    crop: Optional[Sequence[float]] = None,
) -> Path:
    """Full pipeline from file paths to a saved output file (brief §4).

    Streams one image at a time (load → cover-scale → adjustments →
    placement → fold), so memory stays bounded regardless of image count;
    ``crop`` (``(x0, y0, x1, y1)`` fractions of the canvas) is applied to the
    finished composite.  Output location:

    * ``out_path`` given — write exactly there (format inferred from its
      suffix unless ``out_format`` overrides it);
    * otherwise ``blend_<mode>_<YYYYMMDD-HHMMSS>.<ext>`` (brief §4.4) in
      ``out_dir`` (default: current working directory), format
      ``out_format`` (default ``"tiff"``).

    Returns the path of the written file.
    """
    _check_count(len(paths))
    adjs = _resolve_adjustment_list(adjustments, len(paths))
    if crop is not None:
        _validate_crop(crop)
    # Fail fast and by name if a source file has gone (e.g. an external drive
    # was disconnected since the images were added) — before any slow work.
    bs_io.check_sources(paths)

    # Pass 1: cheap size probe to pick target dims (smallest by area, §4.3).
    sizes = [bs_io.probe_size(p) for p in paths]
    target = geometry.target_dimensions(sizes)

    # Pass 2: stream the fold.
    fold = BlendFold(mode, params)
    for path, adj in zip(paths, adjs):
        image = bs_io.load_image(path)
        placed, mask = _prepare_layer(geometry.cover_scale(image, target), adj, target)
        del image
        fold.push(placed, opacity=adj.opacity, mask=mask)
    composite = apply_crop(fold.result(), crop)

    if out_path is not None:
        destination = Path(out_path)
        return bs_io.save_image(composite, destination, format=out_format)
    fmt = (out_format or "tiff").lower()
    directory = Path(out_dir) if out_dir is not None else Path.cwd()
    destination = directory / bs_io.default_filename(mode, fmt)
    return bs_io.save_image(composite, destination, format=fmt)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _check_count(n: int) -> None:
    if not MIN_IMAGES <= n <= MAX_IMAGES:
        raise ValueError(
            f"A blend takes {MIN_IMAGES}–{MAX_IMAGES} images, got {n}"
        )


def _resolve_adjustment_list(
    adjustments: Sequence[AdjustmentsLike] | None, n: int
) -> list[Adjustments]:
    if adjustments is None:
        return [adj_mod.DEFAULT_ADJUSTMENTS] * n
    if len(adjustments) != n:
        raise ValueError(
            f"adjustments must have one entry per image ({n}), got {len(adjustments)}"
        )
    return [Adjustments.from_mapping(a) for a in adjustments]


def _coverage_bool(mask: Optional[np.ndarray],
                   image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Coverage mask → bool (H, W, 1), or ``None`` if absent / all covered."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    if m.shape != image_shape[:2] + (1,):
        raise ValueError(
            f"mask shape {np.asarray(mask).shape} does not match image "
            f"{image_shape[:2]} (expected (H, W, 1))"
        )
    covered = m > 0.5
    return None if covered.all() else covered


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Validate shape and coerce dtype to float32 without copying if
    already float32."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)
