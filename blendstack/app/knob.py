"""Rotary knob + small "rack" UI helpers (synth-hardware look).

:class:`Knob` replaces the old ``SliderRow``.  It works in RAW INTEGER steps
(``minimum`` .. ``maximum``) mapped onto a real value via ``scale``
(``real = raw * scale``), exactly like ``SliderRow`` did, so migration is
mechanical::

    Knob("EXPOSURE", -300, 300, scale=0.01, decimals=2, suffix=" EV")

Interaction: vertical (or horizontal) drag, Shift = fine, mouse wheel, arrow /
Page / Home / End keys, double-click resets to ``default``.

API summary
-----------
* ``value() -> float`` / ``set_value(v)`` (silent) / ``set_value_from_user(v)``
  (sets and emits ``valueChanged`` if it changed - what a drag does)
* ``raw()`` / ``set_raw(i)`` (silent) / ``set_raw_from_user(i)``
* ``reset()`` (emits if the value changed), ``matches(v)``
* ``valueChanged(float)`` in real units
* ``is_bipolar()``, ``is_dragging()``, ``set_dial_size(px)``

The module also exports the tiny shared helpers the panels use: the segmented
"pill" button group (:func:`make_segmented`), section titles / vertical rules,
and :func:`contrast_pairs` listing the colour pairs this module hard-wires so
the self-test can verify them with :func:`theme.contrast_ratio`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from . import theme

__all__ = [
    "Knob",
    "make_segmented",
    "segment_stylesheet",
    "make_section_title",
    "make_vrule",
    "refresh_layouts",
    "ghost_button_stylesheet",
    "contrast_pairs",
]

# ----------------------------------------------------------------- shared bits


def contrast_pairs() -> list[tuple[str, str, str]]:
    """(name, foreground, background) pairs hard-wired in this module."""
    return [
        ("knob label on panel", theme.TEXT_DIM, theme.PANEL),
        ("knob readout on pill", theme.GOLD[0], theme.INSET),
        ("knob readout (dragging) on pill", theme.ACCENT_HI, theme.INSET),
        ("knob label (disabled) on panel", theme.TEXT_DIM, theme.PANEL),
        ("knob readout (disabled) on pill", theme.TEXT_DIM, theme.INSET),
        ("segment text selected", theme.TEXT_ON_GOLD, theme.GOLD[0]),
        ("segment text unselected", theme.TEXT, theme.PANEL_HI),
        ("segment text hover", theme.TEXT, theme.BORDER),
        ("segment text disabled", theme.TEXT_DIM, theme.PANEL_HI),
        ("ghost button text", theme.TEXT, theme.PANEL),
        ("ghost button hover text", theme.GOLD[0], theme.PANEL),
        ("ghost button disabled text", theme.TEXT_DIM, theme.PANEL),
    ]


def segment_stylesheet(position: str) -> str:
    """QSS for one segment of a pill group; ``position`` is 'first', 'last',
    'middle' or 'only'.  Selected = gold fill + dark text; otherwise PANEL_HI
    with light text."""
    r = "9px"
    tl = bl = r if position in ("first", "only") else "0px"
    tr = br = r if position in ("last", "only") else "0px"
    left = "1px" if position in ("first", "only") else "0px"
    return f"""
QPushButton {{
    background: {theme.PANEL_HI}; color: {theme.TEXT};
    border: 1px solid {theme.BORDER_HI}; border-left-width: {left};
    border-top-left-radius: {tl}; border-bottom-left-radius: {bl};
    border-top-right-radius: {tr}; border-bottom-right-radius: {br};
    padding: 0px 8px; font-size: 9px; font-weight: 700; letter-spacing: 0.8px; }}
QPushButton:hover {{ background: {theme.BORDER}; color: {theme.TEXT}; border-color: {theme.BORDER_HI}; }}
QPushButton:checked {{ background: {theme.GOLD[0]}; color: {theme.TEXT_ON_GOLD};
    border-color: {theme.GOLD[1]}; }}
QPushButton:checked:hover {{ background: {theme.ACCENT_HI}; color: {theme.TEXT_ON_GOLD}; }}
QPushButton:disabled {{ color: {theme.TEXT_DIM}; border-color: {theme.BORDER}; }}
QPushButton:checked:disabled {{ background: {theme.GOLD[3]}; color: {theme.TEXT}; border-color: {theme.GOLD[3]}; }}
"""


def ghost_button_stylesheet() -> str:
    """Compact secondary button used for RESET actions in the rack."""
    return f"""
QPushButton {{ background: transparent; color: {theme.TEXT};
    border: 1px solid {theme.BORDER_HI}; border-radius: 9px;
    padding: 0px 10px; font-size: 9px; font-weight: 700; letter-spacing: 0.8px; }}
QPushButton:hover {{ border-color: {theme.GOLD[0]}; color: {theme.GOLD[0]}; }}
QPushButton:pressed {{ background: {theme.INSET}; }}
QPushButton:disabled {{ color: {theme.TEXT_DIM}; border-color: {theme.BORDER}; }}
"""


def make_segmented(
    parent: QWidget, labels: Sequence[str], height: int = 18
) -> tuple[QWidget, list[QPushButton], QButtonGroup]:
    """A horizontal pill group of exclusive, checkable buttons.

    Returns ``(container, buttons, group)``; button ``i`` has group id ``i``.
    The buttons are ordinary ``QPushButton``s (``setChecked`` / ``isChecked``
    / ``toggled`` all work)."""
    box = QWidget(parent)
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    group = QButtonGroup(box)
    group.setExclusive(True)
    buttons: list[QPushButton] = []
    n = len(labels)
    for i, text in enumerate(labels):
        b = QPushButton(text, box)
        b.setCheckable(True)
        b.setFocusPolicy(Qt.NoFocus)
        b.setFixedHeight(height)
        b.setCursor(Qt.PointingHandCursor)
        pos = "only" if n == 1 else "first" if i == 0 else "last" if i == n - 1 else "middle"
        b.setStyleSheet(segment_stylesheet(pos))
        group.addButton(b, i)
        buttons.append(b)
        lay.addWidget(b)
    box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return box, buttons, group


def refresh_layouts(root: QWidget) -> None:
    """Drop every cached layout / size-hint cache under ``root`` so a
    ``sizeHint()`` read right after a size change (dial size, plot size) is
    fresh instead of waiting for the event loop."""
    for lay in root.findChildren(QLayout):
        lay.invalidate()
    for w in root.findChildren(QWidget):
        w.updateGeometry()
    if root.layout() is not None:
        root.layout().invalidate()
        root.layout().activate()
    root.updateGeometry()


def make_section_title(text: str, parent: Optional[QWidget] = None) -> QLabel:
    """Gold, letter-spaced section title."""
    label = QLabel(text.upper(), parent)
    label.setProperty("role", "title")
    return label


def make_vrule(parent: Optional[QWidget] = None) -> QFrame:
    """A thin vertical divider between rack sections."""
    rule = QFrame(parent)
    rule.setObjectName("rackRule")
    rule.setFixedWidth(1)
    rule.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    rule.setStyleSheet(f"QFrame#rackRule {{ background: {theme.BORDER}; border: none; }}")
    return rule


# ------------------------------------------------------------------------ Knob


class _SliderShim:
    """Tiny ``QSlider``-like facade (``setValue`` / ``value`` / ``minimum`` /
    ``maximum`` in RAW steps) so older code that poked ``row.slider`` keeps
    working.  ``setValue`` behaves like a user edit (emits on change)."""

    def __init__(self, knob: "Knob") -> None:
        self._k = knob

    def setValue(self, raw: int) -> None:  # noqa: N802
        self._k.set_raw_from_user(raw)

    def value(self) -> int:
        return self._k.raw()

    def minimum(self) -> int:
        return self._k.minimum()

    def maximum(self) -> int:
        return self._k.maximum()


class Knob(QWidget):
    """A rotary knob with a label and a numeric readout pill underneath."""

    #: Default dial diameters (px).
    NORMAL = 52
    COMPACT = 44

    #: Vertical drag distance (px) for the full range.
    DRAG_RANGE_PX = 200.0
    #: Wheel angle delta that counts as one notch.
    WHEEL_NOTCH = 120

    valueChanged = Signal(float)  # noqa: N815 (Qt signal naming)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        scale: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        default: Optional[float] = None,
        parent: Optional[QWidget] = None,
        *,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        if maximum <= minimum:
            raise ValueError("Knob needs maximum > minimum")
        self._label = label
        self._min = int(minimum)
        self._max = int(maximum)
        self._scale = float(scale)
        self._decimals = int(decimals)
        self._suffix = suffix
        self._bipolar = self._min < 0 < self._max
        if default is None:
            default = 0.0 if self._min <= 0 <= self._max else self._min * self._scale
        self._default_raw = self._clamp_raw(round(float(default) / self._scale))
        self._raw = self._clamp_raw(0 if self._min <= 0 <= self._max else self._min)
        self._dial = int(size if size is not None else self.NORMAL)

        self._hover = False
        self._dragging = False
        self._kbd_focus = False
        self._last_pos = QPointF()
        self._acc = 0.0
        self._wheel_acc = 0

        self._label_font = theme.label_font(8, bold=True)
        self._label_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.7)
        self._mono = theme.mono_font(9, bold=True)

        self.slider = _SliderShim(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.SizeVerCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAccessibleName(label)
        self._refresh_tooltip()

    # ---------------------------------------------------------------- value API

    def minimum(self) -> int:
        return self._min

    def maximum(self) -> int:
        return self._max

    def scale(self) -> float:
        return self._scale

    def label(self) -> str:
        return self._label

    def default(self) -> float:
        return self._real(self._default_raw)

    def is_bipolar(self) -> bool:
        """True if the range spans 0 (arc drawn from 12 o'clock outward)."""
        return self._bipolar

    def is_dragging(self) -> bool:
        return self._dragging

    def raw(self) -> int:
        return self._raw

    def value(self) -> float:
        return self._real(self._raw)

    def _real(self, raw: int) -> float:
        return round(raw * self._scale, 9)

    def _clamp_raw(self, raw: int) -> int:
        return max(self._min, min(self._max, int(raw)))

    def _to_raw(self, value: float) -> int:
        value = float(value)
        if not math.isfinite(value):
            return self._raw
        return self._clamp_raw(round(value / self._scale))

    def matches(self, value: float) -> bool:
        """True if ``value`` lands on the knob's current step (so the knob
        already shows it unchanged)."""
        value = float(value)
        if not math.isfinite(value):
            return False
        return self._raw == round(value / self._scale)

    def set_raw(self, raw: int) -> None:
        """Set the raw step silently (no signal)."""
        self._apply_raw(raw, emit=False)

    def set_raw_from_user(self, raw: int) -> None:
        """Set the raw step and emit ``valueChanged`` if it changed."""
        self._apply_raw(raw, emit=True)

    def set_value(self, value: float) -> None:
        """Set the real value silently (rounded to the nearest step)."""
        self._apply_raw(self._to_raw(value), emit=False)

    def set_value_from_user(self, value: float) -> None:
        """Set the real value and emit if it changed (what a drag does)."""
        self._apply_raw(self._to_raw(value), emit=True)

    def reset(self) -> None:
        """Back to the default value (emits if it changed)."""
        self._apply_raw(self._default_raw, emit=True)

    def _apply_raw(self, raw: int, emit: bool) -> bool:
        raw = self._clamp_raw(raw)
        if raw == self._raw:
            return False
        self._raw = raw
        self._refresh_tooltip()
        self.update()
        if emit:
            self.valueChanged.emit(self.value())
        return True

    # ------------------------------------------------------------------ display

    def format_value(self, raw: Optional[int] = None) -> str:
        v = self._real(self._raw if raw is None else raw)
        v = round(v, self._decimals) + 0.0  # avoid "-0.00"
        text = f"{v:+.{self._decimals}f}" if self._min < 0 else f"{v:.{self._decimals}f}"
        return text + self._suffix

    def _refresh_tooltip(self) -> None:
        self.setToolTip(f"{self._label}: {self.format_value()} — double-click to reset")

    def dial_size(self) -> int:
        return self._dial

    def set_dial_size(self, px: int) -> None:
        px = max(24, int(px))
        if px != self._dial:
            self._dial = px
            self.updateGeometry()
            self.update()

    # -- geometry -----------------------------------------------------------

    _TOP = 3
    _LABEL_H = 11
    _PILL_H = 15

    def _pill_width(self) -> int:
        fm = QFontMetrics(self._mono)
        texts = {self.format_value(self._min), self.format_value(self._max),
                 self.format_value(self._default_raw)}
        return max(fm.horizontalAdvance(t) for t in texts) + 12

    def _label_width(self) -> int:
        return QFontMetrics(self._label_font).horizontalAdvance(self._label.upper()) + 4

    def sizeHint(self) -> QSize:  # noqa: N802
        w = max(self._dial + 14, self._label_width(), self._pill_width())
        h = self._TOP + self._dial + 3 + self._LABEL_H + 2 + self._PILL_H
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    # ------------------------------------------------------------------ painting

    def _t(self) -> float:
        return (self._raw - self._min) / float(self._max - self._min)

    @staticmethod
    def _angle_deg(t: float) -> float:
        """Clockwise degrees from 12 o'clock for fraction ``t`` (-135..+135)."""
        return -135.0 + 270.0 * t

    def _arc(self, p: QPainter, rect: QRectF, theta0: float, theta1: float) -> None:
        """Draw an arc between two clockwise-from-12 angles."""
        start = int(round((90.0 - theta0) * 16))
        span = int(round(-(theta1 - theta0) * 16))
        if span:
            p.drawArc(rect, start, span)

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        enabled = self.isEnabled()
        D = float(self._dial)
        s = D / 52.0
        cx = self.width() / 2.0
        cy = self._TOP + D / 2.0
        c = QPointF(cx, cy)
        t = self._t()
        theta = self._angle_deg(t)
        hot = enabled and self._dragging

        # --- focus ring -------------------------------------------------------
        if enabled and self.hasFocus() and self._kbd_focus:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.color(theme.GOLD[0]), 1.5))
            p.drawEllipse(c, D / 2.0 - 0.8, D / 2.0 - 0.8)

        # --- tick marks ---------------------------------------------------------
        zero_t = (0 - self._min) / float(self._max - self._min) if self._bipolar else 0.0
        lo_t, hi_t = (min(t, zero_t), max(t, zero_t))
        for k in range(11):
            tk = k / 10.0
            a = math.radians(self._angle_deg(tk))
            sx, sy = math.sin(a), -math.cos(a)
            major = k in (0, 5, 10)
            r0 = (22.6 if not major else 22.2) * s
            r1 = 24.4 * s
            lit = enabled and lo_t - 1e-9 <= tk <= hi_t + 1e-9 and (t > 0 or self._bipolar)
            if self._bipolar and k == 5:
                col = theme.color(theme.GREEN[0] if enabled else theme.GREEN[3])
                r0, r1, width = 21.6 * s, 24.8 * s, 1.8
            else:
                col = theme.color(theme.GOLD[3] if lit else theme.BORDER_HI)
                width = 1.2 if major else 1.0
            p.setPen(QPen(col, width, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx + sx * r0, cy + sy * r0), QPointF(cx + sx * r1, cy + sy * r1))

        # --- recessed bezel ------------------------------------------------------
        bez = 21.0 * s
        ring = QColor(theme.BORDER_HI)
        if enabled and self._hover:
            ring = ring.lighter(150)
        if hot:
            ring = theme.color(theme.GOLD[2])
        p.setBrush(theme.color(theme.INSET))
        p.setPen(QPen(ring, 1.3))
        p.drawEllipse(c, bez, bez)

        # --- value track + arc ---------------------------------------------------
        rt = 17.6 * s
        arc_rect = QRectF(cx - rt, cy - rt, 2 * rt, 2 * rt)
        track_w = 3.4 * s
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(theme.color(theme.BORDER), track_w, Qt.SolidLine, Qt.RoundCap))
        self._arc(p, arc_rect, -135.0, 135.0)
        arc_col = theme.GOLD[0] if enabled else theme.ACCENT_LO
        a0, a1 = (0.0, theta) if self._bipolar else (-135.0, theta)
        if abs(a1 - a0) > 0.01:
            glow = theme.color(arc_col, 60 if enabled else 0)
            if enabled:
                p.setPen(QPen(glow, track_w + 4.0 * s, Qt.SolidLine, Qt.RoundCap))
                self._arc(p, arc_rect, a0, a1)
            p.setPen(QPen(theme.color(arc_col), track_w, Qt.SolidLine, Qt.RoundCap))
            self._arc(p, arc_rect, a0, a1)
        elif self._bipolar:
            # a tiny gold nub at the centre so "0" still reads as a value
            p.setPen(QPen(theme.color(arc_col), track_w, Qt.SolidLine, Qt.RoundCap))
            self._arc(p, arc_rect, -1.2, 1.2)

        # --- knob body --------------------------------------------------------------
        rb = 13.2 * s
        p.setPen(Qt.NoPen)
        p.setBrush(theme.color(theme.INSET, 170))
        p.drawEllipse(QPointF(cx, cy + 1.6 * s), rb + 1.2 * s, rb + 1.2 * s)
        grad = QRadialGradient(QPointF(cx - 0.25 * rb, cy - 0.35 * rb), rb * 1.45)
        grad.setColorAt(0.0, theme.color(theme.PANEL_HI).lighter(125 if hot or self._hover else 112))
        grad.setColorAt(1.0, theme.color(theme.INSET))
        bevel = QLinearGradient(QPointF(cx, cy - rb), QPointF(cx, cy + rb))
        bevel.setColorAt(0.0, theme.color(theme.BORDER_HI).lighter(130))
        bevel.setColorAt(1.0, theme.color(theme.BORDER))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QBrush(bevel), 1.2))
        p.drawEllipse(c, rb, rb)

        # --- pointer ------------------------------------------------------------------
        a = math.radians(theta)
        sx, sy = math.sin(a), -math.cos(a)
        white = theme.color(theme.TEXT if enabled else theme.TEXT_DIM)
        tip = theme.color(theme.GOLD[0] if enabled else theme.ACCENT_LO)
        pw = 2.3 * s
        p.setPen(QPen(white, pw, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + sx * 0.12 * rb, cy + sy * 0.12 * rb),
                   QPointF(cx + sx * 0.62 * rb, cy + sy * 0.62 * rb))
        p.setPen(QPen(tip, pw, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + sx * 0.58 * rb, cy + sy * 0.58 * rb),
                   QPointF(cx + sx * 0.90 * rb, cy + sy * 0.90 * rb))

        # --- label --------------------------------------------------------------------
        label_top = self._TOP + D + 3
        p.setFont(self._label_font)
        p.setPen(theme.color(theme.TEXT_DIM if not enabled or not hot else theme.TEXT))
        fm = QFontMetrics(self._label_font)
        text = fm.elidedText(self._label.upper(), Qt.ElideRight, int(self.width()))
        p.drawText(QRectF(0, label_top, self.width(), self._LABEL_H), Qt.AlignCenter, text)

        # --- readout pill -------------------------------------------------------------
        pill_w = float(min(self._pill_width(), self.width() - 2))
        pill = QRectF(cx - pill_w / 2.0, label_top + self._LABEL_H + 2, pill_w, self._PILL_H)
        pill = pill.adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(theme.color(theme.INSET))
        p.setPen(QPen(theme.color(theme.GOLD[2] if hot else theme.BORDER), 1.0))
        p.drawRoundedRect(pill, 4, 4)
        p.setFont(self._mono)
        if not enabled:
            p.setPen(theme.color(theme.TEXT_DIM))
        elif hot:
            p.setPen(theme.color(theme.ACCENT_HI))
        else:
            p.setPen(theme.color(theme.GOLD[0]))
        p.drawText(pill, Qt.AlignCenter, self.format_value())
        p.end()

    # --------------------------------------------------------------------- events

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802
        self._kbd_focus = event.reason() != Qt.MouseFocusReason
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.update()
        super().focusOutEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._dragging = True
        self._last_pos = event.position()
        self._acc = float(self._raw)
        self.setFocus(Qt.MouseFocusReason)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging or not (event.buttons() & Qt.LeftButton):
            event.ignore()
            return
        pos = event.position()
        move = -(pos.y() - self._last_pos.y()) + (pos.x() - self._last_pos.x())
        self._last_pos = pos
        per_px = (self._max - self._min) / self.DRAG_RANGE_PX
        fine = 0.1 if event.modifiers() & Qt.ShiftModifier else 1.0
        self._acc = max(float(self._min), min(float(self._max), self._acc + move * per_px * fine))
        self._apply_raw(int(round(self._acc)), emit=True)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.reset()
            self.update()
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event) -> None:  # noqa: N802
        d = event.angleDelta()
        dy = d.y() or d.x()  # Shift+wheel arrives as a horizontal delta on macOS
        if not dy:
            event.ignore()
            return
        self._wheel_acc += dy
        steps = int(self._wheel_acc / self.WHEEL_NOTCH)
        if steps:
            self._wheel_acc -= steps * self.WHEEL_NOTCH
            mult = 10 if event.modifiers() & Qt.ShiftModifier else 1
            self._apply_raw(self._raw + steps * mult, emit=True)
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        step = {Qt.Key_Up: 1, Qt.Key_Right: 1, Qt.Key_Down: -1, Qt.Key_Left: -1,
                Qt.Key_PageUp: 10, Qt.Key_PageDown: -10}.get(key)
        if step is not None:
            self._kbd_focus = True
            self._apply_raw(self._raw + step, emit=True)
            self.update()
            event.accept()
        elif key == Qt.Key_Home:
            self._apply_raw(self._min, emit=True)
            event.accept()
        elif key == Qt.Key_End:
            self._apply_raw(self._max, emit=True)
            event.accept()
        else:
            super().keyPressEvent(event)
