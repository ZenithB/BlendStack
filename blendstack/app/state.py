"""Document model for the standalone app (project brief §5).

Holds the ordered image list (order = fold order, first = base image), the
per-image :class:`~blendstack.core.adjustments.Adjustments`, and the global
blend settings (mode + parameters + output format).  Emits Qt signals on
every change so the UI and the preview controller can react:

* :attr:`DocumentState.images_changed`      — add / remove / reorder / clear.
* :attr:`DocumentState.adjustments_changed` — one image's own settings
  changed (carries the entry id, so the preview cache can invalidate only
  that image's adjusted proxy).
* :attr:`DocumentState.blend_changed`       — mode or a global parameter
  changed (only the fold needs re-running).

On add, each image immediately gets a **proxy** downscaled to
≤ :data:`PROXY_LONG_EDGE` px on the long edge (brief §5 live preview),
resampled with the core's Lanczos path; the full-resolution pixels are then
released — exports re-read the files (``engine.blend_files`` streams).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PySide6.QtCore import QObject, Signal

from blendstack.core import engine, geometry
from blendstack.core import io as bs_io
from blendstack.core.adjustments import Adjustments

__all__ = ["PROXY_LONG_EDGE", "ImageEntry", "AddReport", "DocumentState"]

#: Preview proxies are downscaled to at most this long-edge size (brief §5).
PROXY_LONG_EDGE = 1440

_ids = itertools.count(1)

PathLike = Union[str, Path]


def make_proxy(image: np.ndarray) -> np.ndarray:
    """Proportionally downscale to ≤ :data:`PROXY_LONG_EDGE` px long edge.

    Uses the core geometry path (Lanczos cover + centre-crop); because the
    target preserves the aspect ratio the "crop" is at most a rounding
    pixel.  Images already small enough pass through untouched.
    """
    height, width = image.shape[:2]
    long_edge = max(width, height)
    if long_edge <= PROXY_LONG_EDGE:
        return image
    scale = PROXY_LONG_EDGE / long_edge
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return geometry.conform(image, target)


@dataclass
class ImageEntry:
    """One image in the document: identity, proxy pixels and settings."""

    entry_id: int
    path: Path
    full_size: tuple[int, int]  # (width, height) of the file on disk
    proxy: np.ndarray           # float32 RGB (H, W, 3), ≤1440 px long edge
    adjustments: Adjustments = field(default_factory=Adjustments)


@dataclass
class AddReport:
    """Outcome of :meth:`DocumentState.add_paths`."""

    added: list[ImageEntry] = field(default_factory=list)
    refused_cap: list[Path] = field(default_factory=list)   # over the 20 cap
    errors: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refused_cap and not self.errors


class DocumentState(QObject):
    """Ordered images + per-image adjustments + global blend settings."""

    images_changed = Signal()
    adjustments_changed = Signal(int)  # entry_id
    blend_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._entries: list[ImageEntry] = []
        self._mode: str = engine.mode_names()[0]
        self._params: dict[str, Any] = engine.get_mode(self._mode).default_params()
        self.output_format: str = "tiff"  # part of preset "output settings"

    # -- read access --------------------------------------------------------

    @property
    def entries(self) -> list[ImageEntry]:
        """The ordered image list (fold order; first = base). Do not mutate."""
        return list(self._entries)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def entry(self, entry_id: int) -> Optional[ImageEntry]:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def entry_ids(self) -> list[int]:
        return [e.entry_id for e in self._entries]

    # -- structural edits ----------------------------------------------------

    def add_paths(
        self,
        paths: Sequence[PathLike],
        adjustments: Optional[Sequence[Adjustments]] = None,
        _defer_signal: bool = False,
    ) -> AddReport:
        """Load files, build proxies and append entries (fold order).

        Enforces the :data:`engine.MAX_IMAGES` cap (brief §8: a 21st image
        is refused) and reports unreadable/unsupported files individually.
        """
        report = AddReport()
        for i, raw_path in enumerate(paths):
            path = Path(raw_path).expanduser()
            try:
                path = path.resolve()
            except OSError:
                pass
            if len(self._entries) >= engine.MAX_IMAGES:
                report.refused_cap.extend(
                    Path(p).expanduser() for p in paths[i:]
                )
                break
            if path.suffix.lower() not in bs_io.SUPPORTED_INPUT_EXTENSIONS:
                report.errors.append((path, "unsupported file type"))
                continue
            try:
                image = bs_io.load_image(path)
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                report.errors.append((path, str(exc) or type(exc).__name__))
                continue
            adj = (
                adjustments[i]
                if adjustments is not None and i < len(adjustments)
                else Adjustments()
            )
            entry = ImageEntry(
                entry_id=next(_ids),
                path=path,
                full_size=(image.shape[1], image.shape[0]),
                proxy=make_proxy(image),
                adjustments=adj,
            )
            self._entries.append(entry)
            report.added.append(entry)
        if report.added and not _defer_signal:
            self.images_changed.emit()
        return report

    def remove(self, entry_ids: Sequence[int]) -> None:
        """Remove the given entries (ignores unknown ids)."""
        doomed = set(entry_ids)
        kept = [e for e in self._entries if e.entry_id not in doomed]
        if len(kept) != len(self._entries):
            self._entries = kept
            self.images_changed.emit()

    def reorder(self, entry_ids: Sequence[int]) -> None:
        """Reorder to match ``entry_ids`` (must be a permutation)."""
        by_id = {e.entry_id: e for e in self._entries}
        if sorted(entry_ids) != sorted(by_id):
            raise ValueError("reorder() requires a permutation of current ids")
        new_order = [by_id[i] for i in entry_ids]
        if new_order != self._entries:
            self._entries = new_order
            self.images_changed.emit()

    def clear(self) -> None:
        if self._entries:
            self._entries = []
            self.images_changed.emit()

    # -- per-image settings ---------------------------------------------------

    def set_adjustments(self, entry_id: int, adjustments: Adjustments) -> None:
        entry = self.entry(entry_id)
        if entry is not None and entry.adjustments != adjustments:
            entry.adjustments = adjustments
            self.adjustments_changed.emit(entry_id)

    # -- global blend settings -------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode != self._mode:
            spec = engine.get_mode(mode)  # validate early
            self._mode = mode
            # Reset params to the new mode's defaults so stale softness/bias/
            # basis (or any other mode's params) never leak across a switch.
            # New modes declare no params → this becomes {}.
            self._params = spec.default_params()
            self.blend_changed.emit()

    def set_param(self, name: str, value: Any) -> None:
        if name not in self._params:
            raise ValueError(f"Unknown blend parameter '{name}'")
        if self._params[name] != value:
            self._params[name] = value
            self.blend_changed.emit()

    # -- preset support (brief §5 Presets) --------------------------------------

    def restore(
        self,
        mode: str,
        params: dict[str, Any],
        output_format: str,
        images: Sequence[tuple[Path, Adjustments]],
    ) -> AddReport:
        """Replace the whole document from preset data.

        Missing/unreadable files come back in the report; everything else
        loads (brief §5: load what it can).
        """
        engine.get_mode(mode)  # validate before touching state
        self._entries = []
        self._mode = mode
        self._params = engine.get_mode(mode).resolve_params(params)
        self.output_format = output_format
        report = self.add_paths(
            [p for p, _ in images],
            adjustments=[a for _, a in images],
            _defer_signal=True,
        )
        self.blend_changed.emit()
        self.images_changed.emit()
        return report
