"""Graphical tone-curve editor for one image (RGB master + R/G/B channels).

:class:`CurveEditor` edits the four control-point lists of
:class:`~blendstack.core.adjustments.Adjustments` (``curve_master``,
``curve_red``, ``curve_green``, ``curve_blue``).  The curve is drawn from
:func:`~blendstack.core.adjustments.curve_lut`, i.e. with exactly the same
monotone-cubic interpolation the engine applies.

Interaction (all on the square plot):

* click empty plot area  -> add a control point there and start dragging it
* drag a point           -> move it (x kept strictly between its neighbours,
  y clamped to 0..1; the two end points may move freely but their x stays
  between 0 / 1 and the adjacent point)
* right-click or double-click a point -> remove it (never below 2 points)
* a point is "hit" when the mouse is within :data:`HIT_RADIUS` px of it.

Every edit emits :attr:`CurveEditor.curves_changed` (continuously while
dragging); :meth:`CurveEditor.set_curves` is silent.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import IDENTITY_CURVE, curve_lut

__all__ = ["CurveEditor", "CHANNELS", "HIT_RADIUS", "MIN_X_GAP"]

#: Channel order == the order of ``curves()`` / ``set_curves()``.
CHANNELS = ("RGB", "R", "G", "B")

#: Pixel radius within which the mouse "hits" a control point.
HIT_RADIUS = 9.0
#: Drawn control-point radius (px).
POINT_RADIUS = 5.0
#: Minimum x distance between two neighbouring control points.
MIN_X_GAP = 0.004

_CHANNEL_COLOURS = (None, QColor(225, 60, 60), QColor(50, 175, 70), QColor(70, 120, 235))

_Points = list  # list[tuple[float, float]]


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _sanitise(points: Sequence[Sequence[float]]) -> _Points:
    """Clamp to 0..1, sort by x, drop points closer than the gap to the
    previous one; fall back to the identity curve if fewer than 2 remain."""
    try:
        pts = sorted(
            (
                (_clamp01(float(p[0])), _clamp01(float(p[1])))
                for p in points
                if math.isfinite(float(p[0])) and math.isfinite(float(p[1]))
            ),
            key=lambda p: p[0],
        )
    except (TypeError, ValueError, IndexError):
        pts = []
    out: _Points = []
    for p in pts:
        if out and not p[0] > out[-1][0]:
            continue
        out.append(p)
    return out if len(out) >= 2 else [tuple(p) for p in IDENTITY_CURVE]


def _is_identity(points: Sequence[Sequence[float]]) -> bool:
    return (
        points[0][0] == 0.0
        and points[-1][0] == 1.0
        and all(x == y for x, y in points)
    )


class _CurvePlot(QWidget):
    """The square plot: painting and mouse handling for a CurveEditor."""

    _MARGIN = 10

    def __init__(self, editor: "CurveEditor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._drag: Optional[int] = None
        self._hover: Optional[int] = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    # -- geometry ------------------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return w

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(260, 260)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(220, 220)

    def plot_rect(self) -> QRectF:
        m = self._MARGIN
        side = max(10.0, min(self.width(), self.height()) - 2 * m)
        return QRectF(
            (self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side
        )

    def to_pixel(self, x: float, y: float) -> QPointF:
        r = self.plot_rect()
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    def from_pixel(self, p: QPointF) -> tuple[float, float]:
        r = self.plot_rect()
        return ((p.x() - r.left()) / r.width(), (r.bottom() - p.y()) / r.height())

    def hit_test(self, pos: QPointF) -> Optional[int]:
        """Index of the nearest control point within HIT_RADIUS px, or None."""
        best, best_d = None, HIT_RADIUS
        for i, (x, y) in enumerate(self._editor._pts[self._editor._channel]):
            q = self.to_pixel(x, y)
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d <= best_d:
                best, best_d = i, d
        return best

    # -- painting ------------------------------------------------------------

    def _channel_colour(self, ch: int) -> QColor:
        c = _CHANNEL_COLOURS[ch]
        return QColor(c) if c is not None else QColor(self.palette().windowText().color())

    def _curve_path(self, points: Sequence[Sequence[float]]) -> QPainterPath:
        n = 513
        lut = curve_lut(tuple(tuple(p) for p in points), n)
        path = QPainterPath()
        for i in range(n):
            q = self.to_pixel(i / (n - 1), float(lut[i]))
            if i == 0:
                path.moveTo(q)
            else:
                path.lineTo(q)
        return path

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = self.palette()
        r = self.plot_rect()
        grid = QColor(pal.mid().color())
        text = QColor(pal.windowText().color())

        p.fillRect(r, pal.base())
        grid.setAlpha(110)
        p.setPen(QPen(grid, 1.0))
        for k in (0.25, 0.5, 0.75):
            a, b = self.to_pixel(k, 0.0), self.to_pixel(k, 1.0)
            p.drawLine(a, b)
            a, b = self.to_pixel(0.0, k), self.to_pixel(1.0, k)
            p.drawLine(a, b)
        # identity reference diagonal
        diag = QColor(text)
        diag.setAlpha(70)
        pen = QPen(diag, 1.0, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(self.to_pixel(0, 0), self.to_pixel(1, 1))
        # frame
        frame = QColor(pal.mid().color())
        p.setPen(QPen(frame, 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)

        ed = self._editor
        # faint curves of the other channels (only if non-identity)
        for ch in range(4):
            if ch == ed._channel or _is_identity(ed._pts[ch]):
                continue
            col = self._channel_colour(ch)
            col.setAlpha(85)
            p.setPen(QPen(col, 1.3))
            p.drawPath(self._curve_path(ed._pts[ch]))
        # selected channel
        col = self._channel_colour(ed._channel)
        p.setPen(QPen(col, 2.0))
        p.drawPath(self._curve_path(ed._pts[ed._channel]))
        for i, (x, y) in enumerate(ed._pts[ed._channel]):
            active = i in (self._hover, self._drag)
            radius = POINT_RADIUS + (1.0 if active else 0.0)
            p.setPen(QPen(text, 1.2))
            fill = QColor(col)
            fill = fill.lighter(130) if active else fill
            p.setBrush(fill)
            p.drawEllipse(self.to_pixel(x, y), radius, radius)
        p.end()

    # -- mouse -----------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        ed = self._editor
        hit = self.hit_test(pos)
        if event.button() == Qt.RightButton:
            if hit is not None:
                ed.remove_point(ed._channel, hit)
                self._hover = self.hit_test(pos)
                self._announce(self._hover)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        if hit is None:
            x, y = self.from_pixel(pos)
            hit = ed.add_point(ed._channel, x, y)
            hit = hit if hit >= 0 else None
        self._drag = hit
        self._hover = hit
        self._announce(hit)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            hit = self.hit_test(event.position())
            if hit is not None:
                self._drag = None
                self._editor.remove_point(self._editor._channel, hit)
                self._hover = self.hit_test(event.position())
                self._announce(self._hover)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        ed = self._editor
        if self._drag is not None and (event.buttons() & Qt.LeftButton):
            x, y = self.from_pixel(pos)
            ed.move_point(ed._channel, self._drag, x, y)
            self._hover = self._drag
            self._announce(self._drag)
            event.accept()
            return
        hit = self.hit_test(pos)
        if hit != self._hover:
            self._hover = hit
            self.update()
        self.setCursor(Qt.PointingHandCursor if hit is not None else Qt.CrossCursor)
        self._announce(hit)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag = None
            self._hover = self.hit_test(event.position())
            self.update()
        event.accept()

    def leaveEvent(self, _event: object) -> None:  # noqa: N802
        if self._drag is None:
            self._hover = None
            self._announce(None)
            self.update()

    def _announce(self, index: Optional[int]) -> None:
        ed = self._editor
        pts = ed._pts[ed._channel]
        if index is None or not (0 <= index < len(pts)):
            ed._set_readout(None)
        else:
            x, y = pts[index]
            ed._set_readout((x, y))


class CurveEditor(QWidget):
    """Four-channel tone-curve editor (RGB master, R, G, B)."""

    #: Emitted after any user (or programmatic add/move/remove/reset) edit.
    curves_changed = Signal()

    _HINT = "Click to add a point; double-click or right-click a point to remove"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pts: list[_Points] = [[tuple(p) for p in IDENTITY_CURVE] for _ in range(4)]
        self._channel = 0

        # channel selector (compact segmented buttons)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._tabs: list[QPushButton] = []
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(0)
        for i, name in enumerate(CHANNELS):
            b = QPushButton(name, self)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet(
                "QPushButton { padding: 2px 6px; border: 1px solid palette(mid);"
                " background: palette(button); }"
                "QPushButton:checked { background: palette(highlight);"
                " color: palette(highlighted-text); }"
            )
            self._group.addButton(b, i)
            self._tabs.append(b)
            tab_row.addWidget(b, 1)
        self._tabs[0].setChecked(True)
        self._group.idClicked.connect(self._on_tab)

        self.plot = _CurvePlot(self)

        self._readout = QLabel(self._HINT, self)
        self._readout.setWordWrap(True)
        small = self._readout.font()
        small.setPointSizeF(max(8.0, small.pointSizeF() * 0.85))
        self._readout.setFont(small)
        self._readout.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # reserve two lines so swapping hint <-> readout never shifts the layout
        self._readout.setMinimumHeight(2 * self._readout.fontMetrics().lineSpacing())
        self._readout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.reset_channel_button = QPushButton("Reset channel", self)
        self.reset_all_button = QPushButton("Reset all curves", self)
        self.reset_channel_button.clicked.connect(self.reset_channel)
        self.reset_all_button.clicked.connect(self.reset_all)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.reset_channel_button)
        btn_row.addWidget(self.reset_all_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(tab_row)
        layout.addWidget(self.plot)
        layout.addWidget(self._readout)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ state

    def curves(self) -> tuple:
        """``(master, red, green, blue)`` as tuples of ``(x, y)`` float tuples."""
        return tuple(
            tuple((float(x), float(y)) for x, y in pts) for pts in self._pts
        )

    def set_curves(self, master, red, green, blue) -> None:
        """Replace all four curves silently (never emits)."""
        self._pts = [_sanitise(c) for c in (master, red, green, blue)]
        self.plot._drag = None
        self.plot._hover = None
        self._set_readout(None)
        self.plot.update()

    def channel(self) -> int:
        return self._channel

    def set_channel(self, idx: int) -> None:
        """Select the channel being edited (0 RGB, 1 R, 2 G, 3 B)."""
        idx = max(0, min(3, int(idx)))
        self._channel = idx
        self._tabs[idx].setChecked(True)
        self.plot._drag = None
        self.plot._hover = None
        self._set_readout(None)
        self.plot.update()

    def _on_tab(self, idx: int) -> None:
        self.set_channel(idx)

    # ------------------------------------------------------------- editing API

    def _changed(self) -> None:
        self.plot.update()
        self.curves_changed.emit()

    def add_point(self, channel: int, x: float, y: float) -> int:
        """Insert a control point; returns its index, or -1 if it would sit
        within :data:`MIN_X_GAP` of an existing point's x."""
        pts = self._pts[channel]
        x, y = _clamp01(float(x)), _clamp01(float(y))
        if any(abs(px - x) < MIN_X_GAP for px, _ in pts):
            return -1
        idx = sum(1 for px, _ in pts if px < x)
        pts.insert(idx, (x, y))
        self._changed()
        return idx

    def move_point(self, channel: int, index: int, x: float, y: float) -> bool:
        """Move a point; x is clamped strictly between its neighbours (end
        points: between 0/1 and the adjacent point), y to 0..1.  Returns
        True if anything changed."""
        pts = self._pts[channel]
        if not 0 <= index < len(pts):
            return False
        lo = pts[index - 1][0] + MIN_X_GAP if index > 0 else 0.0
        hi = pts[index + 1][0] - MIN_X_GAP if index < len(pts) - 1 else 1.0
        nx = min(max(float(x), lo), hi)
        ny = _clamp01(float(y))
        if (nx, ny) == pts[index]:
            return False
        pts[index] = (nx, ny)
        self._changed()
        return True

    def remove_point(self, channel: int, index: int) -> bool:
        """Delete a point (never leaving fewer than 2).  Returns True if removed."""
        pts = self._pts[channel]
        if len(pts) <= 2 or not 0 <= index < len(pts):
            return False
        del pts[index]
        self._changed()
        return True

    def reset_channel(self) -> None:
        """Reset the selected channel to the identity curve."""
        ident = [tuple(p) for p in IDENTITY_CURVE]
        if self._pts[self._channel] != ident:
            self._pts[self._channel] = ident
            self._changed()

    def reset_all(self) -> None:
        """Reset all four curves to identity."""
        ident = [tuple(p) for p in IDENTITY_CURVE]
        if any(c != ident for c in self._pts):
            self._pts = [list(ident) for _ in range(4)]
            self._changed()

    # ----------------------------------------------------------------- readout

    def _set_readout(self, point: Optional[tuple[float, float]]) -> None:
        if point is None:
            self._readout.setText(self._HINT)
        else:
            self._readout.setText(
                f"In {round(point[0] * 255)} → Out {round(point[1] * 255)}"
            )
