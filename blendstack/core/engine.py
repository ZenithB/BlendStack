"""Pipeline orchestration (project brief §4).

For each render (preview or export)::

    for each image i in user-defined order:
        load -> float32 RGB 0..1        (RAW via LibRaw defaults; GIF frame 0;
                                         alpha flattened against black)
        geometry: cover-scale + centre-crop to target dims (smallest by area)
        adjustments, fixed order: exposure -> contrast -> saturation -> sharpen
    accumulator = image[0]              (opacity ignored for the first image)
    for each subsequent image i:
        blended = mode.blend(accumulator, image[i], params)
        accumulator = lerp(accumulator, blended, opacity[i])
    clip accumulator to 0..1
    encode to output format / bit depth

Public API for frontends (GUI and GIMP plugin):

* :func:`blend_arrays` — already-loaded float32 arrays in, composite out.
* :func:`blend_files`  — file paths in, saved output file out (streams one
  image at a time, so 20 × 24 MP stays within a small memory budget).
* :func:`conform_images` / :func:`adjust_image` — the geometry and
  adjustment steps exposed separately so a frontend can cache per-image
  intermediate results and re-run only the fold on slider drags (brief §5).
* :class:`BlendFold` — incremental fold, one image at a time.

Linear-light scaffolding (brief §3 design rule 2): if the selected mode
declares ``needs_linear = True``, the fold inputs are linearised with the
sRGB EOTF before folding and the accumulator is re-encoded afterwards.
Both v1 modes declare False (max/min are monotone-invariant, brief §1).

All opacity and mode parameters are in UI units throughout (opacity
0–100 %, softness 0–100, bias −100…+100).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

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
    """

    def __init__(self, mode: str = "canon_bright",
                 params: Mapping[str, Any] | None = None) -> None:
        self._mode: BlendMode = get_mode(mode)
        # Resolve immediately: fills defaults and rejects unknown names.
        self._params: dict[str, Any] = self._mode.resolve_params(params)
        self._accumulator: Optional[np.ndarray] = None
        self._count = 0

    @property
    def count(self) -> int:
        """Number of images pushed so far."""
        return self._count

    def push(self, image: np.ndarray, opacity: float = 100.0) -> None:
        """Fold one image in. ``opacity`` is 0–100 (ignored for the first
        image).  Raises ``ValueError`` past the 20-image cap."""
        if self._count >= MAX_IMAGES:
            raise ValueError(f"Blend cap exceeded: at most {MAX_IMAGES} images")
        image = _as_float_rgb(image)
        if self._mode.needs_linear:
            image = adj_mod.srgb_to_linear(image)
        if self._accumulator is None:
            self._accumulator = image  # first image: opacity ignored (§4)
        else:
            if self._accumulator.shape != image.shape:
                raise ValueError(
                    f"Image shape {image.shape} does not match accumulator "
                    f"{self._accumulator.shape}; run the geometry step first"
                )
            blended = self._mode.blend(self._accumulator, image, self._params)
            k = float(opacity) / 100.0
            if k >= 1.0:
                self._accumulator = blended
            else:
                self._accumulator = self._accumulator + (
                    blended - self._accumulator
                ) * np.float32(k)
        self._count += 1

    def result(self) -> np.ndarray:
        """Re-encode (if the mode ran in linear light), clip to 0–1 and
        return the composite as a new float32 array."""
        if self._accumulator is None:
            raise ValueError("No images have been pushed")
        out = self._accumulator
        if self._mode.needs_linear:
            out = adj_mod.linear_to_srgb(out)
        # np.clip allocates a fresh array, so the accumulator stays untouched.
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def fold_images(
    images: Sequence[np.ndarray],
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
    opacities: Sequence[float] | None = None,
) -> np.ndarray:
    """Fold step only: blend pre-conformed, pre-adjusted images.

    ``opacities`` are 0–100 per image (first entry ignored, default all
    100).  Returns the clipped float32 composite."""
    if opacities is not None and len(opacities) != len(images):
        raise ValueError("opacities must have one entry per image")
    fold = BlendFold(mode, params)
    for i, image in enumerate(images):
        fold.push(image, 100.0 if opacities is None else opacities[i])
    return fold.result()


# --------------------------------------------------------------------------
# High-level API
# --------------------------------------------------------------------------

def blend_arrays(
    images: Sequence[np.ndarray],
    adjustments: Sequence[AdjustmentsLike] | None = None,
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Full pipeline on already-loaded images: geometry → adjustments →
    fold → clip (brief §4).  Returns the float32 composite (H, W, 3), 0–1.

    ``images``      — 2–20 float32 RGB arrays, (H, W, 3), values 0–1.
    ``adjustments`` — optional, one per image: :class:`Adjustments` or a
                      dict with any of ``exposure, contrast, saturation,
                      sharpen_radius, sharpen_amount, opacity`` (UI units);
                      ``None`` entries mean defaults.
    ``mode``        — registered mode name (``"canon_bright"``/``"canon_dark"``).
    ``params``      — mode parameters, e.g. ``{"softness": 20, "bias": -10,
                      "basis": "luminance"}``; omitted keys take defaults.
    """
    _check_count(len(images))
    adjs = _resolve_adjustment_list(adjustments, len(images))
    conformed = conform_images([_as_float_rgb(im) for im in images])
    fold = BlendFold(mode, params)
    for image, adj in zip(conformed, adjs):
        fold.push(adjust_image(image, adj), opacity=adj.opacity)
    return fold.result()


def blend_files(
    paths: Sequence[PathLike],
    mode: str = "canon_bright",
    params: Mapping[str, Any] | None = None,
    adjustments: Sequence[AdjustmentsLike] | None = None,
    out_path: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    out_format: Optional[str] = None,
) -> Path:
    """Full pipeline from file paths to a saved output file (brief §4).

    Streams one image at a time (load → geometry → adjustments → fold), so
    memory stays bounded regardless of image count.  Output location:

    * ``out_path`` given — write exactly there (format inferred from its
      suffix unless ``out_format`` overrides it);
    * otherwise ``blend_<mode>_<YYYYMMDD-HHMMSS>.<ext>`` (brief §4.4) in
      ``out_dir`` (default: current working directory), format
      ``out_format`` (default ``"tiff"``).

    Returns the path of the written file.
    """
    _check_count(len(paths))
    adjs = _resolve_adjustment_list(adjustments, len(paths))

    # Pass 1: cheap size probe to pick target dims (smallest by area, §4.3).
    sizes = [bs_io.probe_size(p) for p in paths]
    target = geometry.target_dimensions(sizes)

    # Pass 2: stream the fold.
    fold = BlendFold(mode, params)
    for path, adj in zip(paths, adjs):
        image = bs_io.load_image(path)
        image = geometry.conform(image, target)
        image = adjust_image(image, adj)
        fold.push(image, opacity=adj.opacity)
    composite = fold.result()

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


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Validate shape and coerce dtype to float32 without copying if
    already float32."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)
