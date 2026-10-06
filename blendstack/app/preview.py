"""Live preview rendering (project brief §5, "Live preview" — mandatory).

Threading model
---------------

* :class:`PreviewController` lives on the UI thread.  Any state change calls
  :meth:`PreviewController.request_render`, which (re)starts an ~80 ms
  single-shot debounce ``QTimer``.  When it fires, the controller snapshots
  the document (entry ids, proxy arrays, adjustments, mode, params, solo id,
  crop, "show full canvas"), stamps it with a fresh **generation number** and
  dispatches it to the worker via a queued signal.  The UI thread never runs
  the fold.

* :class:`RenderWorker` lives on a dedicated ``QThread``.  Per layer it runs
  the core layer pipeline on the proxies: ``geometry.cover_scale`` (to the
  smallest proxy by area) -> ``engine.adjust_image`` (on the SCALED image) ->
  ``geometry.place`` (centre + the layer's ``move_x`` / ``move_y`` offset,
  gap = transparent mask) -> ``engine.BlendFold.push(image, opacity, mask)``.
  Between steps it checks whether a newer generation was requested, in which
  case it aborts (stale-render cancellation).

  *Solo* renders ONE layer in isolation (cover-scale -> adjust -> place, the
  uncovered area black, opacity/blend mode ignored).  The result is cropped
  with ``engine.apply_crop`` unless the request asks for the full canvas
  (Move / Crop tool on).

* **Mute**: a muted image (``adjustments.mute``) still counts for the canvas
  size (smallest by area over ALL images, exactly like the core engine, so
  layer positions and the crop never shift when mute is toggled) but is
  skipped in the fold — no scale / adjust work is done for it.  Fewer than 2
  un-muted images → no render (the canvas shows a "Un-mute at least 2 images"
  placeholder).  *Solo* overrides mute: a soloed muted image is shown.

* **Caching** (brief §5): ``cover_scale`` results are cached by
  ``(entry id, target dims)`` and adjusted images by
  ``(entry id, target dims, adj.adjust_key())``.  ``adjust_key`` covers every
  setting that changes the adjusted pixels but NOT opacity / move_x / move_y,
  so dragging a layer on the canvas (and dragging an opacity or global blend
  slider) re-runs only the cheap ``place`` + fold — never the adjustments.
  Arrays are immutable once built, so sharing references across threads is
  safe.  ``scale_calls`` / ``adjust_calls`` count cache misses (test hooks).

The composite handed back is the post-clip float32 accumulator (cropped or
not); the histogram (4×256: R, G, B, Rec.709 luma) is computed from it in the
worker so every preview render updates the histogram (brief §5 "Histogram").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from blendstack.core import engine, geometry
from blendstack.core.adjustments import Adjustments, rec709_luma

from . import canvas_tools, theme
from .canvas_tools import CropEditor
from .state import DocumentState

__all__ = [
    "DEBOUNCE_MS",
    "array_to_qimage",
    "RenderRequest",
    "RenderItem",
    "RenderWorker",
    "PreviewController",
    "PreviewCanvas",
    "PLACEHOLDER_TEXT",
    "MUTE_PLACEHOLDER_TEXT",
]

#: Debounce interval for preview re-renders (brief §5: ~80 ms).
DEBOUNCE_MS = 80

#: Placeholder shown when there are fewer than 2 images.
PLACEHOLDER_TEXT = (
    "Drop 2–20 images here to blend\n"
    "(or use Open in the toolbar)"
)

#: Placeholder shown when >= 2 images are loaded but fewer than 2 are un-muted.
MUTE_PLACEHOLDER_TEXT = (
    "Un-mute at least 2 images to blend\n"
    "(click M on a row, or press Shift+M)"
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
    """One debounced render job, stamped with its generation.

    ``solo_id`` renders just that layer in isolation; ``crop`` is the canvas
    crop (fractions) and is applied unless ``show_full_canvas`` is set (the
    Move / Crop tool shows the whole canvas)."""

    generation: int
    items: tuple[RenderItem, ...]
    mode: str
    params: dict[str, Any]
    solo_id: Optional[int] = None
    crop: Optional[tuple[float, float, float, float]] = None
    show_full_canvas: bool = False


class RenderWorker(QObject):
    """Fold executor living on the render thread (never the UI thread)."""

    #: (generation, composite float32 (H, W, 3), histogram int64 (4, 256),
    #:  full_canvas — True if the frame is the uncropped canvas)
    finished = Signal(int, object, object, bool)
    failed = Signal(int, str)

    def __init__(self, clock: _GenerationClock) -> None:
        super().__init__()
        self._clock = clock
        # (entry_id, target_dims) -> cover-scaled proxy
        self._scaled: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
        # (entry_id, target_dims, adjust_key) -> adjusted (scaled) proxy
        self._adjusted: dict[tuple[int, tuple[int, int], tuple], np.ndarray] = {}
        #: Cache-miss counters (how often the expensive steps really ran).
        self.scale_calls = 0
        self.adjust_calls = 0

    def _stale(self, generation: int) -> bool:
        """True once a newer render has been requested (cancellation)."""
        return generation != self._clock.value

    def _scaled_proxy(
        self, item: RenderItem, target: tuple[int, int]
    ) -> np.ndarray:
        key = (item.entry_id, target)
        cached = self._scaled.get(key)
        if cached is None:
            cached = geometry.cover_scale(item.proxy, target)
            self.scale_calls += 1
            self._scaled[key] = cached
        return cached

    def _adjusted_proxy(
        self, item: RenderItem, target: tuple[int, int]
    ) -> np.ndarray:
        key = (item.entry_id, target, item.adjustments.adjust_key())
        cached = self._adjusted.get(key)
        if cached is None:
            cached = engine.adjust_image(
                self._scaled_proxy(item, target), item.adjustments
            )
            self.adjust_calls += 1
            # Keep only the newest variant per image (slider drags would
            # otherwise accumulate one large array per intermediate value).
            for stale in [k for k in self._adjusted if k[0] == item.entry_id]:
                del self._adjusted[stale]
            self._adjusted[key] = cached
        return cached

    def _placed(
        self, item: RenderItem, target: tuple[int, int]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Cached adjusted layer -> canvas placement (cheap, every render)."""
        adjusted = self._adjusted_proxy(item, target)
        dx, dy = geometry.offset_px(
            item.adjustments.move_x, item.adjustments.move_y, target
        )
        return geometry.place(adjusted, target, dx, dy)

    def _prune(self, live: set[int], target: tuple[int, int]) -> None:
        """Drop cache entries of removed images / outdated canvas sizes."""
        for key in [k for k in self._scaled if k[0] not in live or k[1] != target]:
            del self._scaled[key]
        for key in [k for k in self._adjusted if k[0] not in live or k[1] != target]:
            del self._adjusted[key]

    @Slot(object)
    def render(self, request: RenderRequest) -> None:
        """Run one render job; abort quietly if superseded."""
        generation = request.generation
        if self._stale(generation):
            return
        try:
            sizes = [
                (item.proxy.shape[1], item.proxy.shape[0])
                for item in request.items
            ]
            target = geometry.target_dimensions(sizes)
            self._prune({item.entry_id for item in request.items}, target)

            solo = None
            if request.solo_id is not None:
                solo = next(
                    (i for i in request.items if i.entry_id == request.solo_id),
                    None,
                )
            if solo is not None:
                # One layer in isolation: opacity / blend mode do not apply,
                # uncovered canvas stays black (place() zero-fills it).
                composite, _mask = self._placed(solo, target)
            else:
                fold = engine.BlendFold(request.mode, request.params)
                # Muted images size the canvas (above) but never enter the
                # fold, so they cost no scale / adjust work.
                for item in request.items:
                    if item.adjustments.mute:
                        continue
                    if self._stale(generation):
                        return
                    placed, mask = self._placed(item, target)
                    fold.push(placed, opacity=item.adjustments.opacity, mask=mask)
                if self._stale(generation):
                    return
                composite = fold.result()
            full = request.show_full_canvas or request.crop is None
            if not request.show_full_canvas:
                composite = engine.apply_crop(composite, request.crop)
            if self._stale(generation):
                return
            histogram = compute_histogram(composite)
        except Exception as exc:  # noqa: BLE001 — reported to the UI
            self.failed.emit(generation, str(exc) or type(exc).__name__)
            return
        if self._stale(generation):
            return
        self.finished.emit(generation, composite, histogram, full)


class PreviewController(QObject):
    """UI-thread owner of the debounce timer, generation clock and worker."""

    #: (composite float32 array, histogram (4, 256), full_canvas) — current
    #: generation only.  ``full_canvas`` is True when the frame is the whole
    #: uncropped canvas (Move / Crop tool on, or no crop set).
    preview_ready = Signal(object, object, bool)
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
        self._show_full_canvas = False

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
        state.solo_changed.connect(self.request_render)
        state.crop_changed.connect(self.request_render)

    @property
    def worker(self) -> RenderWorker:
        """The render worker (exposed for cache-behaviour tests)."""
        return self._worker

    @property
    def requested_generation(self) -> int:
        return self._clock.value

    @property
    def show_full_canvas(self) -> bool:
        return self._show_full_canvas

    def set_show_full_canvas(self, on: bool) -> None:
        """Show the whole uncropped canvas (Move / Crop tool) or the crop."""
        if bool(on) != self._show_full_canvas:
            self._show_full_canvas = bool(on)
            self.request_render()

    def request_render(self, *_ignored: object) -> None:
        """Debounced entry point — restart the ~80 ms timer."""
        self._timer.start()

    def _fire(self) -> None:
        entries = self._state.entries
        solo_id = self._state.solo_id
        if solo_id is not None and not any(e.entry_id == solo_id for e in entries):
            solo_id = None
        active = sum(1 for e in entries if not e.adjustments.mute)
        if solo_id is None and active < engine.MIN_IMAGES:
            # Invalidate any in-flight render and show the placeholder.
            self.completed_generation = self._clock.advance()
            self.preview_cleared.emit(
                PLACEHOLDER_TEXT if len(entries) < engine.MIN_IMAGES
                else MUTE_PLACEHOLDER_TEXT
            )
            return
        generation = self._clock.advance()
        request = RenderRequest(
            generation=generation,
            items=tuple(
                RenderItem(e.entry_id, e.proxy, e.adjustments) for e in entries
            ),
            mode=self._state.mode,
            params=self._state.params,
            solo_id=solo_id,
            crop=self._state.crop,
            show_full_canvas=self._show_full_canvas,
        )
        self._dispatch.emit(request)

    @Slot(int, object, object, bool)
    def _on_finished(
        self,
        generation: int,
        composite: np.ndarray,
        histogram: np.ndarray,
        full_canvas: bool,
    ) -> None:
        if generation != self._clock.value:
            return  # superseded while in flight — discard stale result
        self.completed_generation = generation
        self.preview_ready.emit(composite, histogram, full_canvas)

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
    """Centre preview canvas: scaled-to-fit, aspect preserved (brief §5).

    Also hosts the Move / Crop tool interaction (see :mod:`canvas_tools`):

    * With the tool enabled **and** a full-canvas frame displayed
      (:meth:`interactive`), press-and-drag emits :attr:`drag_started` /
      :attr:`drag_moved` (total delta as canvas fractions since the press,
      derived from the letterboxed display rectangle) / :attr:`drag_finished`.
    * Double-click enters crop mode: a :class:`~canvas_tools.CropEditor`
      overlay on the full canvas; drags then edit the rectangle instead.
    * A gold "SOLO — name" pill is drawn top-left while a label is set
      (:meth:`set_solo_label`).

    The canvas is a recessed "screen": near-black inset, a 1 px frame, corner
    registration marks and a soft vignette; placeholders are centred with a
    small drawn glyph.
    """

    drag_started = Signal()
    drag_moved = Signal(float, float)   # total (dx, dy) as canvas fractions
    drag_finished = Signal()
    crop_mode_changed = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._composite: Optional[np.ndarray] = None
        self._frame_full = False
        self._placeholder = PLACEHOLDER_TEXT
        self._banner_rect = QRectF()
        self._tool = False
        self._applied_crop: Optional[tuple[float, float, float, float]] = None
        self._solo_label: Optional[str] = None
        self._editor: Optional[CropEditor] = None
        self._move_press: Optional[QPointF] = None
        self._move_rect = QRectF()
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)

    # -- frame ---------------------------------------------------------------

    def set_composite(self, composite: np.ndarray, full_canvas: bool = True) -> None:
        """Convert the float accumulator to a pixmap and repaint.

        ``full_canvas`` says whether the frame is the whole uncropped
        canvas; mouse interaction is only accepted for such frames."""
        self._composite = composite
        self._pixmap = QPixmap.fromImage(array_to_qimage(composite))
        self._frame_full = bool(full_canvas)
        self._update_cursor()
        self.update()

    def clear(self, message: str = PLACEHOLDER_TEXT) -> None:
        self._abort_move()
        self._pixmap = None
        self._composite = None
        self._frame_full = False
        self._placeholder = message
        if self._editor is not None:
            self.exit_crop_mode()
        self._update_cursor()
        self.update()

    def placeholder_text(self) -> str:
        """The placeholder text shown while no frame is displayed."""
        return self._placeholder

    def solo_banner_rect(self) -> QRectF:
        """Where the SOLO pill was last painted (widget pixels; empty if none)."""
        return QRectF(self._banner_rect)

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def composite(self) -> Optional[np.ndarray]:
        """The float frame currently displayed (read-only; tests/inspection)."""
        return self._composite

    def frame_is_full(self) -> bool:
        """True if the displayed frame is the full, uncropped canvas."""
        return self.has_image() and self._frame_full

    def display_rect(self) -> QRectF:
        """Letterboxed rectangle (widget pixels) the frame is drawn in."""
        if not self.has_image():
            return QRectF()
        return canvas_tools.display_rect(self._pixmap.size(), self.size())

    # -- tool state ----------------------------------------------------------------

    def set_tool_enabled(self, on: bool) -> None:
        """Turn the Move / Crop tool on/off.  Turning it off aborts a drag in
        progress and leaves crop mode (discarding the edit)."""
        on = bool(on)
        if on == self._tool:
            return
        self._tool = on
        if not on:
            self._abort_move()
            if self._editor is not None:
                self.exit_crop_mode()
        self._update_cursor()
        self.update()

    def tool_enabled(self) -> bool:
        return self._tool

    def interactive(self) -> bool:
        """Mouse input is accepted: tool on AND a full-canvas frame shown."""
        return self._tool and self.frame_is_full()

    def set_applied_crop(self, crop: Optional[tuple[float, float, float, float]]) -> None:
        """The crop stored in the document (drawn as an outline while the
        tool is on and no crop edit is running)."""
        self._applied_crop = None if crop is None else tuple(crop)  # type: ignore[assignment]
        self.update()

    def set_solo_label(self, text: Optional[str]) -> None:
        """Show/hide the SOLO banner (``text`` is the file name)."""
        label = None if not text else f"SOLO — {text}"
        if label != self._solo_label:
            self._solo_label = label
            self.update()

    def solo_label(self) -> Optional[str]:
        return self._solo_label

    # -- crop mode --------------------------------------------------------------------

    def crop_mode(self) -> bool:
        return self._editor is not None

    def enter_crop_mode(
        self, initial: Optional[tuple[float, float, float, float]] = None
    ) -> bool:
        """Start editing the crop rectangle (initially ``initial``, else the
        applied crop, else the canvas inset ~8 %).  Needs an interactive
        frame; returns whether crop mode is now active."""
        if self._editor is not None:
            return True
        if not self.interactive():
            return False
        self._abort_move()
        self._editor = CropEditor(initial or self._applied_crop)
        self._update_cursor()
        self.update()
        self.crop_mode_changed.emit(True)
        return True

    def exit_crop_mode(self) -> None:
        """Leave crop mode, discarding the edit (the window stores a crop
        only through Apply, reading :meth:`crop_rect` first)."""
        if self._editor is None:
            return
        self._editor = None
        self._update_cursor()
        self.update()
        self.crop_mode_changed.emit(False)

    def crop_rect(self) -> Optional[tuple[float, float, float, float]]:
        """The crop being edited (fractions), or ``None`` outside crop mode."""
        return None if self._editor is None else tuple(self._editor.crop)

    def crop_handle_point(self, handle: str) -> QPoint:
        """Widget position of a crop handle (test hook / accessibility)."""
        if self._editor is None:
            raise RuntimeError("not in crop mode")
        return self._editor.handle_point(handle, self.display_rect()).toPoint()

    # -- mouse -------------------------------------------------------------------------

    def _abort_move(self) -> None:
        if self._move_press is not None:
            self._move_press = None
            self.drag_finished.emit()

    def _update_cursor(self, pos: Optional[QPointF] = None) -> None:
        if not self.interactive():
            self.unsetCursor()
            return
        if self._editor is not None:
            handle = None
            if self._editor.dragging:
                handle = self._editor.active_handle
            elif pos is not None:
                handle = self._editor.hit_test(pos, self.display_rect())
            self.setCursor(canvas_tools.cursor_for_handle(handle))
        elif self._move_press is not None:
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton or not self.interactive():
            super().mousePressEvent(event)
            return
        pos = event.position()
        rect = self.display_rect()
        if self._editor is not None:
            handle = self._editor.hit_test(pos, rect)
            if handle is not None:
                self._editor.begin_drag(handle, pos, rect)
                self._update_cursor()
        else:
            self._move_press = pos
            self._move_rect = rect
            self._update_cursor()
            self.drag_started.emit()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if not self.interactive():
            super().mouseMoveEvent(event)
            return
        if self._editor is not None:
            if self._editor.dragging:
                self._editor.drag_to(pos, self.display_rect())
                self.update()
            self._update_cursor(pos)
        elif self._move_press is not None:
            rect = self._move_rect
            self.drag_moved.emit(
                (pos.x() - self._move_press.x()) / rect.width(),
                (pos.y() - self._move_press.y()) / rect.height(),
            )
        else:
            self._update_cursor(pos)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._editor is not None and self._editor.dragging:
            self._editor.end_drag()
            self._update_cursor(event.position())
            self.update()
        elif self._move_press is not None:
            self._move_press = None
            self._update_cursor(event.position())
            self.drag_finished.emit()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self.interactive():
            if self._editor is None:
                self.enter_crop_mode()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -- painting -------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._banner_rect = QRectF()
        self._paint_backdrop(painter)
        if self._pixmap is None or self._pixmap.isNull():
            self._paint_placeholder(painter)
        else:
            rect = self.display_rect()
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(theme.color(theme.BORDER_HI), 1.0))
            painter.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))
            if self._tool and self._frame_full:
                self._paint_crop_overlay(painter, rect)
            if self._solo_label:
                self._paint_banner(painter, rect)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(theme.color(theme.BORDER), 1.0))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()

    def _paint_backdrop(self, painter: QPainter) -> None:
        """Recessed screen: inset fill, soft vignette, registration marks."""
        r = self.rect()
        painter.fillRect(r, theme.color(theme.INSET))
        # Vignette — darkens the corners a touch, like a CRT bezel.
        radius = max(r.width(), r.height()) * 0.75
        grad = QRadialGradient(QPointF(r.center()), radius)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.65, QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, QColor(0, 0, 0, 90))
        painter.fillRect(r, grad)
        # Registration marks: corner brackets + edge-centre ticks.
        painter.setPen(QPen(theme.color(theme.BORDER_HI, 150), 1.0))
        m, n = 7.0, 12.0
        w, h = float(r.width()), float(r.height())
        for x, sx in ((m, 1), (w - m, -1)):
            for y, sy in ((m, 1), (h - m, -1)):
                painter.drawLine(QPointF(x, y), QPointF(x + sx * n, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + sy * n))
        t = 5.0
        painter.drawLine(QPointF(w / 2, 1), QPointF(w / 2, 1 + t))
        painter.drawLine(QPointF(w / 2, h - 1), QPointF(w / 2, h - 1 - t))
        painter.drawLine(QPointF(1, h / 2), QPointF(1 + t, h / 2))
        painter.drawLine(QPointF(w - 1, h / 2), QPointF(w - 1 - t, h / 2))

    def _paint_placeholder(self, painter: QPainter) -> None:
        """Centred glyph + dim text (>= 13 px)."""
        muted = self._placeholder == MUTE_PLACEHOLDER_TEXT
        font = theme.label_font(13)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        lines = self._placeholder.split("\n")
        line_h = metrics.height() + 2
        glyph = 56.0
        gap = 16.0
        total = glyph + gap + line_h * len(lines)
        top = (self.height() - total) / 2.0
        cx = self.width() / 2.0
        self._paint_glyph(painter, QRectF(cx - glyph / 2, top, glyph, glyph), muted)
        painter.setPen(theme.color(theme.TEXT_DIM))
        y = top + glyph + gap
        for k, line in enumerate(lines):
            f = QFont(font)
            if k == 0:
                f.setBold(True)
            painter.setFont(f)
            painter.drawText(
                QRectF(0, y, self.width(), line_h), Qt.AlignHCenter | Qt.AlignVCenter, line
            )
            y += line_h

    @staticmethod
    def _paint_glyph(painter: QPainter, box: QRectF, muted: bool) -> None:
        """Icon-like glyph: two stacked frames (layers); with a slash when the
        placeholder is about muted images."""
        pen = QPen(theme.color(theme.TEXT_DIM), 2.0)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        back = QRectF(box.left() + 10, box.top() + 4, box.width() - 14, box.height() - 18)
        front = QRectF(box.left() + 2, box.top() + 14, box.width() - 14, box.height() - 18)
        painter.drawRoundedRect(back, 4, 4)
        painter.setBrush(theme.color(theme.INSET))
        painter.drawRoundedRect(front, 4, 4)
        painter.setBrush(Qt.NoBrush)
        if muted:
            painter.setPen(QPen(theme.color(theme.PURPLE[0]), 4.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(box.left() + 2, box.bottom() - 2),
                             QPointF(box.right() - 2, box.top() + 2))
        else:
            c = front.center()
            painter.setPen(QPen(theme.color(theme.GOLD[0]), 2.5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(c.x() - 6, c.y()), QPointF(c.x() + 6, c.y()))
            painter.drawLine(QPointF(c.x(), c.y() - 6), QPointF(c.x(), c.y() + 6))

    def _paint_crop_overlay(self, painter: QPainter, rect: QRectF) -> None:
        gold = theme.color(theme.GOLD[0])
        if self._editor is not None:
            crop = self._editor.crop
            dim = 140  # ~55 % black outside the crop
        elif self._applied_crop is not None and not canvas_tools.is_full_crop(
            self._applied_crop
        ):
            crop = self._applied_crop
            dim = 90  # an applied crop: lighter dimming
        else:
            return
        x0, y0, x1, y1 = crop
        box = QRectF(
            canvas_tools.fraction_to_widget(x0, y0, rect),
            canvas_tools.fraction_to_widget(x1, y1, rect),
        )
        outside = QPainterPath()
        outside.addRect(rect)
        inner = QPainterPath()
        inner.addRect(box)
        painter.fillPath(outside.subtracted(inner), QColor(0, 0, 0, dim))
        painter.setBrush(Qt.NoBrush)
        painter.setRenderHint(QPainter.Antialiasing, False)
        if self._editor is None:
            # Applied crop (tool on, not editing): dashed gold outline.
            pen = QPen(gold, 1.5, Qt.DashLine)
            pen.setDashPattern([5.0, 3.0])
            painter.setPen(pen)
            painter.drawRect(box)
            painter.setRenderHint(QPainter.Antialiasing, True)
            return
        painter.setPen(QPen(gold, 1.5))
        painter.drawRect(box)
        # Rule-of-thirds guides (translucent text colour).
        painter.setPen(QPen(theme.color(theme.TEXT, 100), 1.0))
        for k in (1, 2):
            gx = box.left() + box.width() * k / 3.0
            gy = box.top() + box.height() * k / 3.0
            painter.drawLine(QPointF(gx, box.top()), QPointF(gx, box.bottom()))
            painter.drawLine(QPointF(box.left(), gy), QPointF(box.right(), gy))
        # Handles: 4 corners + 4 edge midpoints — gold squares, dark outline.
        painter.setPen(QPen(theme.color(theme.INSET), 1.5))
        painter.setBrush(gold)
        half = 4.5
        for name in canvas_tools.HANDLES:
            p = self._editor.handle_point(name, rect)
            painter.drawRect(QRectF(p.x() - half, p.y() - half, 2 * half, 2 * half))
        painter.setRenderHint(QPainter.Antialiasing, True)

    def _paint_banner(self, painter: QPainter, rect: QRectF) -> None:
        """Gold pill: "SOLO — filename" in dark text."""
        font = theme.label_font(11, bold=True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text = metrics.elidedText(
            self._solo_label or "", Qt.ElideMiddle, max(60, int(rect.width()) - 56)
        )
        pad_x = 14
        box = QRectF(
            rect.left() + 10,
            rect.top() + 10,
            metrics.horizontalAdvance(text) + 2 * pad_x,
            metrics.height() + 10,
        )
        self._banner_rect = box
        painter.setPen(QPen(theme.color(theme.GOLD[1]), 1.0))
        painter.setBrush(theme.color(theme.GOLD[0]))
        painter.drawRoundedRect(box, box.height() / 2, box.height() / 2)
        painter.setPen(theme.color(theme.TEXT_ON_GOLD))
        painter.drawText(box, Qt.AlignCenter, text)
