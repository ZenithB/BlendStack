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
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import IDENTITY_CURVE, curve_lut

from . import theme
from .knob import ghost_button_stylesheet, make_segmented

__all__ = ["CurveEditor", "CHANNELS", "HIT_RADIUS", "MIN_X_GAP", "contrast_pairs"]

#: Channel order == the order of ``curves()`` / ``set_curves()``.
CHANNELS = ("RGB", "R", "G", "B")

#: Pixel radius within which the mouse "hits" a control point.
HIT_RADIUS = 9.0
#: Drawn control-point radius (px).
POINT_RADIUS = 4.5
#: Minimum x distance between two neighbouring control points.
MIN_X_GAP = 0.004

#: Plot edge length (px) in the normal and the compact rack layout.
PLOT_SIZE = 170
PLOT_SIZE_COMPACT = 146

_CHANNEL_COLOURS = (theme.GOLD[0], theme.CHANNEL_R, theme.CHANNEL_G, theme.CHANNEL_B)


def contrast_pairs() -> list[tuple[str, str, str]]:
    """(name, foreground, background) pairs hard-wired in this module."""
    return [
        ("curve readout idle on panel", theme.TEXT_DIM, theme.PANEL),
        ("curve readout active on panel", theme.GOLD[0], theme.PANEL),
        ("curve channel R on plot", theme.CHANNEL_R, theme.INSET),
        ("curve channel G on plot", theme.CHANNEL_G, theme.INSET),
        ("curve channel B on plot", theme.CHANNEL_B, theme.INSET),
        ("curve master on plot", theme.GOLD[0], theme.INSET),
    ]

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

    _MARGIN = 5

    def __init__(self, editor: "CurveEditor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._drag: Optional[int] = None
        self._hover: Optional[int] = None
        self._side = PLOT_SIZE
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setFixedSize(self._side, self._side)
        self.setToolTip(
            "Click to add a point \u2022 drag to move it \u2022 "
            "double-click or right-click a point to remove it"
        )

    # -- geometry ------------------------------------------------------------

    def set_side(self, side: int) -> None:
        self._side = int(side)
        self.setFixedSize(self._side, self._side)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._side, self._side)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._side, self._side)

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
        return theme.color(_CHANNEL_COLOURS[ch])

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
        r = self.plot_rect()
        enabled = self.isEnabled()
        # recessed display
        p.setPen(Qt.NoPen)
        p.setBrush(theme.color(theme.INSET))
        p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 3, 3)
        grid = theme.color(theme.BORDER, 255)
        p.setPen(QPen(grid, 1.0))
        for k in (0.25, 0.5, 0.75):
            a, b = self.to_pixel(k, 0.0), self.to_pixel(k, 1.0)
            p.drawLine(QPointF(round(a.x()) + 0.5, a.y()), QPointF(round(b.x()) + 0.5, b.y()))
            a, b = self.to_pixel(0.0, k), self.to_pixel(1.0, k)
            p.drawLine(QPointF(a.x(), round(a.y()) + 0.5), QPointF(b.x(), round(b.y()) + 0.5))
        # identity reference diagonal (faint)
        p.setPen(QPen(theme.color(theme.TEXT, 55), 1.0, Qt.DashLine))
        p.drawLine(self.to_pixel(0, 0), self.to_pixel(1, 1))
        # frame
        p.setPen(QPen(theme.color(theme.BORDER_HI), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 3, 3)

        ed = self._editor
        # faint curves of the other channels (only if non-identity)
        for ch in range(4):
            if ch == ed._channel or _is_identity(ed._pts[ch]):
                continue
            col = self._channel_colour(ch)
            col.setAlpha(95)
            p.setPen(QPen(col, 1.4))
            p.drawPath(self._curve_path(ed._pts[ch]))
        # selected channel: soft glow + thick curve
        col = self._channel_colour(ed._channel)
        if not enabled:
            col.setAlpha(120)
        path = self._curve_path(ed._pts[ed._channel])
        glow = QColor(col)
        glow.setAlpha(50)
        p.setPen(QPen(glow, 6.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        p.setPen(QPen(col, 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        for i, (x, y) in enumerate(ed._pts[ed._channel]):
            active = i in (self._hover, self._drag)
            q = self.to_pixel(x, y)
            if active:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(theme.color(theme.GOLD[0]), 1.8))
                p.drawEllipse(q, POINT_RADIUS + 3.2, POINT_RADIUS + 3.2)
            p.setPen(QPen(theme.color(theme.TEXT), 1.3))
            p.setBrush(col)
            p.drawEllipse(q, POINT_RADIUS, POINT_RADIUS)
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

    _IDLE = "IN \u2192 OUT"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pts: list[_Points] = [[tuple(p) for p in IDENTITY_CURVE] for _ in range(4)]
        self._channel = 0

        # channel selector: pill group  RGB | R | G | B
        tabs_box, self._tabs, self._group = make_segmented(self, CHANNELS)
        self._tabs[0].setChecked(True)
        self._group.idClicked.connect(self._on_tab)

        self.plot = _CurvePlot(self)

        mono = theme.mono_font(9, bold=True)
        self._readout = QLabel(self._IDLE, self)
        self._readout.setFont(mono)
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._readout.setFixedWidth(QFontMetrics(mono).horizontalAdvance("255 \u2192 255") + 4)
        self._set_readout_style(False)

        self.reset_channel_button = QPushButton("RESET CH", self)
        self.reset_all_button = QPushButton("RESET ALL", self)
        for b, tip in ((self.reset_channel_button, "Reset the selected channel's curve"),
                       (self.reset_all_button, "Reset all four curves")):
            b.setFixedHeight(18)
            b.setStyleSheet(ghost_button_stylesheet())
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
        self.reset_channel_button.clicked.connect(self.reset_channel)
        self.reset_all_button.clicked.connect(self.reset_all)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(tabs_box)
        top_row.addStretch(1)
        top_row.addWidget(self._readout)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        btn_row.addWidget(self.reset_channel_button)
        btn_row.addWidget(self.reset_all_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top_row)
        layout.addWidget(self.plot, 0, Qt.AlignHCenter)
        layout.addLayout(btn_row)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_compact(self, compact: bool) -> None:
        """Shrink / restore the plot (rack on a narrow or short window)."""
        self.plot.set_side(PLOT_SIZE_COMPACT if compact else PLOT_SIZE)
        self.updateGeometry()

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

    def _set_readout_style(self, active: bool) -> None:
        self._readout.setStyleSheet(
            f"color: {theme.GOLD[0] if active else theme.TEXT_DIM}; background: transparent;"
        )

    def _set_readout(self, point: Optional[tuple[float, float]]) -> None:
        if point is None:
            self._readout.setText(self._IDLE)
            self._set_readout_style(False)
        else:
            self._readout.setText(f"{round(point[0] * 255)} \u2192 {round(point[1] * 255)}")
            self._set_readout_style(True)
