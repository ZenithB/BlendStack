"""Reusable interaction logic for the preview canvas (Move / Crop tool).

Everything here is plain geometry on **fractions of the full canvas**
(0..1 on both axes), independent of widgets and painting, so it can be unit
tested and reused:

* :class:`CropEditor` — the crop rectangle with 8 handles (4 corners, 4
  edges) plus "drag inside to move".  It clamps to the canvas and keeps a
  minimum size expressed in *displayed* pixels.
* :func:`display_rect` — the letterboxed rectangle an image of a given size
  occupies inside a widget (aspect preserved, centred).
* :func:`widget_to_fraction` / :func:`fraction_to_widget` — mapping between
  widget pixels and canvas fractions through that rectangle.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt

__all__ = [
    "MIN_CROP_PX",
    "HANDLE_TOLERANCE_PX",
    "INITIAL_CROP_INSET",
    "HANDLES",
    "CropEditor",
    "display_rect",
    "widget_to_fraction",
    "fraction_to_widget",
    "inset_crop",
    "is_full_crop",
    "cursor_for_handle",
]

#: Smallest crop rectangle edge, in displayed (widget) pixels.
MIN_CROP_PX = 16.0
#: Distance within which a press grabs a handle / edge, in widget pixels.
HANDLE_TOLERANCE_PX = 9.0
#: Initial crop rectangle inset (each side) when no crop exists yet.
INITIAL_CROP_INSET = 0.08

#: Handle names: corners, edges.
HANDLES = ("tl", "tr", "bl", "br", "t", "b", "l", "r")

Crop = tuple[float, float, float, float]


def inset_crop(inset: float = INITIAL_CROP_INSET) -> Crop:
    """The whole canvas inset by ``inset`` on every side."""
    return (inset, inset, 1.0 - inset, 1.0 - inset)


def is_full_crop(crop: Optional[Sequence[float]], eps: float = 1e-4) -> bool:
    """True if ``crop`` is ``None`` or covers the whole canvas (a no-op)."""
    if crop is None:
        return True
    x0, y0, x1, y1 = crop
    return x0 <= eps and y0 <= eps and x1 >= 1.0 - eps and y1 >= 1.0 - eps


def display_rect(image_size: QSize, widget_size: QSize) -> QRectF:
    """Where an image of ``image_size`` is drawn in a widget of
    ``widget_size``: aspect preserved (scaled to fit), centred."""
    if image_size.isEmpty() or widget_size.isEmpty():
        return QRectF()
    fitted = image_size.scaled(widget_size, Qt.KeepAspectRatio)
    x = (widget_size.width() - fitted.width()) / 2.0
    y = (widget_size.height() - fitted.height()) / 2.0
    return QRectF(x, y, fitted.width(), fitted.height())


def widget_to_fraction(pos: QPointF, rect: QRectF) -> tuple[float, float]:
    """Widget position → canvas fractions through the displayed ``rect``
    (not clamped: positions outside the image map outside 0..1)."""
    return (
        (pos.x() - rect.x()) / rect.width(),
        (pos.y() - rect.y()) / rect.height(),
    )


def fraction_to_widget(fx: float, fy: float, rect: QRectF) -> QPointF:
    """Canvas fractions → widget position through the displayed ``rect``."""
    return QPointF(rect.x() + fx * rect.width(), rect.y() + fy * rect.height())


def cursor_for_handle(handle: Optional[str]) -> Qt.CursorShape:
    """The cursor shape that goes with a handle name (``None`` → arrow)."""
    return {
        "tl": Qt.SizeFDiagCursor,
        "br": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor,
        "bl": Qt.SizeBDiagCursor,
        "l": Qt.SizeHorCursor,
        "r": Qt.SizeHorCursor,
        "t": Qt.SizeVerCursor,
        "b": Qt.SizeVerCursor,
        "move": Qt.SizeAllCursor,
    }.get(handle or "", Qt.ArrowCursor)


class CropEditor:
    """Crop rectangle (fractions of the full canvas) + drag state machine.

    Usage: ``handle = editor.hit_test(pos, rect)`` on hover, then
    ``editor.begin_drag(handle, pos, rect)`` on press, ``editor.drag_to(pos,
    rect)`` while moving and ``editor.end_drag()`` on release.  ``rect`` is
    always the *displayed* image rectangle in widget pixels, which converts
    pixel positions to fractions and expresses the minimum size.
    """

    def __init__(self, crop: Optional[Sequence[float]] = None) -> None:
        self.crop: Crop = tuple(crop) if crop is not None else inset_crop()  # type: ignore[assignment]
        self._handle: Optional[str] = None
        self._start_frac: tuple[float, float] = (0.0, 0.0)
        self._start_crop: Crop = self.crop

    # -- state -----------------------------------------------------------------

    @property
    def dragging(self) -> bool:
        return self._handle is not None

    @property
    def active_handle(self) -> Optional[str]:
        return self._handle

    def reset(self, crop: Optional[Sequence[float]] = None) -> None:
        self.crop = tuple(crop) if crop is not None else inset_crop()  # type: ignore[assignment]
        self._handle = None

    # -- geometry helpers ----------------------------------------------------------

    def rect_px(self, rect: QRectF) -> QRectF:
        """The crop rectangle in widget pixels."""
        x0, y0, x1, y1 = self.crop
        tl = fraction_to_widget(x0, y0, rect)
        br = fraction_to_widget(x1, y1, rect)
        return QRectF(tl, br)

    def handle_point(self, handle: str, rect: QRectF) -> QPointF:
        """Widget position of a named handle (``move`` = the centre)."""
        x0, y0, x1, y1 = self.crop
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        fx = {"tl": x0, "l": x0, "bl": x0, "t": xm, "b": xm, "move": xm,
              "tr": x1, "r": x1, "br": x1}[handle]
        fy = {"tl": y0, "t": y0, "tr": y0, "l": ym, "r": ym, "move": ym,
              "bl": y1, "b": y1, "br": y1}[handle]
        return fraction_to_widget(fx, fy, rect)

    def hit_test(self, pos: QPointF, rect: QRectF) -> Optional[str]:
        """Which handle a press at ``pos`` would grab, or ``None``.

        Corners win over edges; edges are grabbed within
        :data:`HANDLE_TOLERANCE_PX` of the edge line (and the span of the
        rectangle, extended by the tolerance); inside the rectangle = move.
        """
        r = self.rect_px(rect)
        tol = HANDLE_TOLERANCE_PX
        # With a tiny rectangle the tolerance must not swallow the interior.
        tol_x = min(tol, max(r.width() / 3.0, 3.0))
        tol_y = min(tol, max(r.height() / 3.0, 3.0))
        near_l = abs(pos.x() - r.left()) <= tol_x
        near_r = abs(pos.x() - r.right()) <= tol_x
        near_t = abs(pos.y() - r.top()) <= tol_y
        near_b = abs(pos.y() - r.bottom()) <= tol_y
        in_x = r.left() - tol_x <= pos.x() <= r.right() + tol_x
        in_y = r.top() - tol_y <= pos.y() <= r.bottom() + tol_y
        if not (in_x and in_y):
            return None
        if near_t and near_l:
            return "tl"
        if near_t and near_r:
            return "tr"
        if near_b and near_l:
            return "bl"
        if near_b and near_r:
            return "br"
        if near_t:
            return "t"
        if near_b:
            return "b"
        if near_l:
            return "l"
        if near_r:
            return "r"
        return "move"

    # -- drag ----------------------------------------------------------------------

    def begin_drag(self, handle: str, pos: QPointF, rect: QRectF) -> None:
        self._handle = handle
        self._start_frac = widget_to_fraction(pos, rect)
        self._start_crop = self.crop

    def end_drag(self) -> None:
        self._handle = None

    def drag_to(self, pos: QPointF, rect: QRectF) -> Crop:
        """Update the crop for a drag now at ``pos``; returns the new crop."""
        if self._handle is None:
            return self.crop
        fx, fy = widget_to_fraction(pos, rect)
        dx = fx - self._start_frac[0]
        dy = fy - self._start_frac[1]
        x0, y0, x1, y1 = self._start_crop
        min_w = min(MIN_CROP_PX / rect.width(), 0.5)
        min_h = min(MIN_CROP_PX / rect.height(), 0.5)
        h = self._handle

        if h == "move":
            w, ht = x1 - x0, y1 - y0
            nx0 = min(max(x0 + dx, 0.0), 1.0 - w)
            ny0 = min(max(y0 + dy, 0.0), 1.0 - ht)
            self.crop = (nx0, ny0, nx0 + w, ny0 + ht)
            return self.crop

        if "l" in h:
            x0 = min(max(x0 + dx, 0.0), x1 - min_w)
        if "r" in h:
            x1 = max(min(x1 + dx, 1.0), x0 + min_w)
        if "t" in h:
            y0 = min(max(y0 + dy, 0.0), y1 - min_h)
        if "b" in h:
            y1 = max(min(y1 + dy, 1.0), y0 + min_h)
        self.crop = (x0, y0, x1, y1)
        return self.crop
