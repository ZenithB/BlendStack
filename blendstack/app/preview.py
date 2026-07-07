"""Live preview rendering (project brief §5, "Live preview" — mandatory).

Threading model
---------------

* :class:`PreviewController` lives on the UI thread.  Any state change calls
  :meth:`PreviewController.request_render`, which (re)starts an ~80 ms
  single-shot debounce ``QTimer``.  When it fires, the controller snapshots
  the document (entry ids, proxy arrays, adjustments, mode, params), stamps
  it with a fresh **generation number** and dispatches it to the worker via
  a queued signal.  The UI thread never runs the fold.

* :class:`RenderWorker` lives on a dedicated ``QThread``.  It conforms the
  proxies to a common size (smallest proxy by area, via the core geometry
  path), applies per-image adjustments, folds with
  :class:`blendstack.core.engine.BlendFold` and computes the composite
  histogram — checking between steps whether a newer generation has been
  requested, in which case it aborts (stale-render cancellation).

* **Caching** (brief §5): the worker caches, per entry id, the conformed
  proxy and the adjusted proxy.  The adjusted-proxy cache key is the
  image's *own* adjustment settings (opacity excluded — it acts at the fold
  step), so dragging a global slider re-runs only the fold and dragging one
  image's exposure re-adjusts only that image.  Proxy arrays are immutable
  once built, so sharing references across threads is safe.

The composite handed back is the post-clip float32 accumulator; the
histogram (4×256: R, G, B, Rec.709 luma) is computed from it in the worker
so every preview render updates the histogram (brief §5 "Histogram").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from blendstack.core import engine, geometry
from blendstack.core.adjustments import Adjustments, rec709_luma

from .state import DocumentState

__all__ = [
    "DEBOUNCE_MS",
    "array_to_qimage",
    "RenderRequest",
    "RenderWorker",
    "PreviewController",
    "PreviewCanvas",
]

#: Debounce interval for preview re-renders (brief §5: ~80 ms).
DEBOUNCE_MS = 80

#: Placeholder shown when there are fewer than 2 images.
PLACEHOLDER_TEXT = (
    "Drop 2–20 images here to blend\n"
    "(or use Open in the toolbar)"
)


def array_to_qimage(arr: np.ndarray) -> QImage:
    """Float 0–1 (H, W, 3) → owned RGB888 ``QImage``.

    The uint8 buffer is made contiguous, ``bytesPerLine`` is passed
    explicitly, and ``.copy()`` detaches the QImage from the NumPy buffer so
    it cannot be garbage-collected out from under Qt (brief §5 hand-off).
    """
    rgb8 = np.ascontiguousarray(
        (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    )
    height, width = rgb8.shape[:2]
    image = QImage(rgb8.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()  # detach from the numpy-owned buffer


def compute_histogram(composite: np.ndarray) -> np.ndarray:
    """256-bin R, G, B and Rec.709 luma histograms of the post-clip
    accumulator (brief §5 "Histogram").  Returns int64 (4, 256)."""
    q = np.clip(composite, 0.0, 1.0)
    bins8 = (q * 255.0).round().astype(np.uint8)
    hist = np.empty((4, 256), dtype=np.int64)
    for c in range(3):
        hist[c] = np.bincount(bins8[..., c].ravel(), minlength=256)
    luma = (np.clip(rec709_luma(q)[..., 0], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    hist[3] = np.bincount(luma.ravel(), minlength=256)
    return hist


class _GenerationClock:
    """Monotonic render-generation counter shared between threads."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def advance(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


@dataclass(frozen=True)
class RenderItem:
    """Immutable snapshot of one image for a render job."""

    entry_id: int
    proxy: np.ndarray
    adjustments: Adjustments


@dataclass(frozen=True)
class RenderRequest:
    """One debounced render job, stamped with its generation."""

    generation: int
    items: tuple[RenderItem, ...]
    mode: str
    params: dict[str, Any]


class RenderWorker(QObject):
    """Fold executor living on the render thread (never the UI thread)."""

    #: (generation, composite float32 (H, W, 3), histogram int64 (4, 256))
    finished = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(self, clock: _GenerationClock) -> None:
        super().__init__()
        self._clock = clock
        # entry_id -> (target_dims, conformed proxy)
        self._conformed: dict[int, tuple[tuple[int, int], np.ndarray]] = {}
        # entry_id -> (target_dims, adjustment key, adjusted proxy)
        self._adjusted: dict[
            int, tuple[tuple[int, int], tuple[float, ...], np.ndarray]
        ] = {}

    def _stale(self, generation: int) -> bool:
        """True once a newer render has been requested (cancellation)."""
        return generation != self._clock.value

    @staticmethod
    def _adjust_key(adj: Adjustments) -> tuple[float, ...]:
        """Cache key: the image's own settings, opacity excluded (opacity
        acts at the fold step and must not invalidate the adjusted proxy)."""
        return (
            adj.exposure,
            adj.contrast,
            adj.saturation,
            adj.sharpen_radius,
            adj.sharpen_amount,
        )

    def _conformed_proxy(
        self, item: RenderItem, target: tuple[int, int]
    ) -> np.ndarray:
        cached = self._conformed.get(item.entry_id)
        if cached is not None and cached[0] == target:
            return cached[1]
        arr = geometry.conform(item.proxy, target)
        self._conformed[item.entry_id] = (target, arr)
        return arr

    def _adjusted_proxy(
        self, item: RenderItem, target: tuple[int, int]
    ) -> np.ndarray:
        key = self._adjust_key(item.adjustments)
        cached = self._adjusted.get(item.entry_id)
        if cached is not None and cached[0] == target and cached[1] == key:
            return cached[2]
        arr = engine.adjust_image(
            self._conformed_proxy(item, target), item.adjustments
        )
        self._adjusted[item.entry_id] = (target, key, arr)
        return arr

    @Slot(object)
    def render(self, request: RenderRequest) -> None:
        """Run one render job; abort quietly if superseded."""
        generation = request.generation
        if self._stale(generation):
            return
        try:
            # Drop caches for removed images.
            live = {item.entry_id for item in request.items}
            for cache in (self._conformed, self._adjusted):
                for entry_id in list(cache):
                    if entry_id not in live:
                        del cache[entry_id]

            sizes = [
                (item.proxy.shape[1], item.proxy.shape[0])
                for item in request.items
            ]
            target = geometry.target_dimensions(sizes)

            fold = engine.BlendFold(request.mode, request.params)
            for item in request.items:
                if self._stale(generation):
                    return
                fold.push(
                    self._adjusted_proxy(item, target),
                    opacity=item.adjustments.opacity,
                )
            if self._stale(generation):
                return
            composite = fold.result()
            histogram = compute_histogram(composite)
        except Exception as exc:  # noqa: BLE001 — reported to the UI
            self.failed.emit(generation, str(exc) or type(exc).__name__)
            return
        if self._stale(generation):
            return
        self.finished.emit(generation, composite, histogram)


class PreviewController(QObject):
    """UI-thread owner of the debounce timer, generation clock and worker."""

    #: (composite float32 array, histogram (4, 256)) — current generation only.
    preview_ready = Signal(object, object)
    #: Emitted instead of a render when fewer than 2 images are loaded.
    preview_cleared = Signal(str)
    #: A render raised; carries the error text.
    render_failed = Signal(str)

    _dispatch = Signal(object)  # internal, queued into the worker thread

    def __init__(self, state: DocumentState, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._clock = _GenerationClock()
        self.completed_generation = 0

        self._worker = RenderWorker(self._clock)
        self._thread = QThread()
        self._thread.setObjectName("blendstack-preview-render")
        self._worker.moveToThread(self._thread)
        self._dispatch.connect(self._worker.render, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.failed.connect(self._on_failed, Qt.QueuedConnection)
        self._thread.start()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._fire)

        state.images_changed.connect(self.request_render)
        state.adjustments_changed.connect(self.request_render)
        state.blend_changed.connect(self.request_render)

    @property
    def requested_generation(self) -> int:
        return self._clock.value

    def request_render(self, *_ignored: object) -> None:
        """Debounced entry point — restart the ~80 ms timer."""
        self._timer.start()

    def _fire(self) -> None:
        entries = self._state.entries
        if len(entries) < engine.MIN_IMAGES:
            # Invalidate any in-flight render and show the placeholder.
            self.completed_generation = self._clock.advance()
            self.preview_cleared.emit(PLACEHOLDER_TEXT)
            return
        generation = self._clock.advance()
        request = RenderRequest(
            generation=generation,
            items=tuple(
                RenderItem(e.entry_id, e.proxy, e.adjustments) for e in entries
            ),
            mode=self._state.mode,
            params=self._state.params,
        )
        self._dispatch.emit(request)

    @Slot(int, object, object)
    def _on_finished(
        self, generation: int, composite: np.ndarray, histogram: np.ndarray
    ) -> None:
        if generation != self._clock.value:
            return  # superseded while in flight — discard stale result
        self.completed_generation = generation
        self.preview_ready.emit(composite, histogram)

    @Slot(int, str)
    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._clock.value:
            return
        self.completed_generation = generation
        self.render_failed.emit(message)

    def stop(self) -> None:
        """Shut the render thread down (call from closeEvent)."""
        self._timer.stop()
        self._clock.advance()  # cancel anything in flight
        self._thread.quit()
        self._thread.wait(3000)


class PreviewCanvas(QWidget):
    """Centre preview canvas: scaled-to-fit, aspect preserved (brief §5)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._placeholder = PLACEHOLDER_TEXT
        self.setMinimumSize(320, 240)

    def set_composite(self, composite: np.ndarray) -> None:
        """Convert the float accumulator to a pixmap and repaint."""
        self._pixmap = QPixmap.fromImage(array_to_qimage(composite))
        self.update()

    def clear(self, message: str = PLACEHOLDER_TEXT) -> None:
        self._pixmap = None
        self._placeholder = message
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 30))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(150, 150, 155))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            painter.end()
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
