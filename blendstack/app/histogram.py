"""Composite histogram widget (project brief §5, "Histogram").

Custom-painted (no matplotlib): 256-bin R, G, B and Rec.709 luma curves
overlaid, computed by the preview worker from the post-clip accumulator and
pushed here after every preview render.  Purpose: judge highlight
accumulation during build-up, matching the R5's composite histogram
workflow.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

__all__ = ["HistogramWidget", "contrast_pairs"]

#: Draw order: R, G, B first, luma on top.
_CHANNEL_COLORS = (
    theme.CHANNEL_R,
    theme.CHANNEL_G,
    theme.CHANNEL_B,
    theme.CHANNEL_L,  # Rec.709 luma
)


def contrast_pairs() -> list[tuple[str, str, str]]:
    """(name, foreground, background) pairs hard-wired in this module."""
    return [
        ("histogram caption on inset", theme.TEXT_DIM, theme.INSET),
        ("histogram red on inset", theme.CHANNEL_R, theme.INSET),
        ("histogram green on inset", theme.CHANNEL_G, theme.INSET),
        ("histogram blue on inset", theme.CHANNEL_B, theme.INSET),
        ("histogram luma on inset", theme.CHANNEL_L, theme.INSET),
    ]


class HistogramWidget(QWidget):
    """Paints a (4, 256) bin-count array as four overlaid curves (R, G, B
    with soft translucent fills, luma as a bright line)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hist: Optional[np.ndarray] = None
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._caption_font = theme.mono_font(8, bold=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, 100)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(120, 56)

    def set_data(self, hist: Optional[np.ndarray]) -> None:
        """``hist`` is int (4, 256) — R, G, B, luma — or None to clear."""
        if hist is not None:
            hist = np.asarray(hist)
            if hist.shape != (4, 256):
                raise ValueError(f"Expected a (4, 256) histogram, got {hist.shape}")
        self._hist = hist
        self.update()

    def has_data(self) -> bool:
        return self._hist is not None and bool(self._hist.any())

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(theme.color(theme.BORDER), 1.0))
        painter.setBrush(theme.color(theme.INSET))
        painter.drawRoundedRect(rect, 4, 4)
        inner = rect.adjusted(1, 1, -1, -1)

        # faint grid: quarter verticals + a mid horizontal
        painter.setPen(QPen(theme.color(theme.BORDER, 150), 1.0))
        for i in (1, 2, 3):
            x = round(inner.left() + inner.width() * i / 4.0) + 0.5
            painter.drawLine(QPointF(x, inner.top() + 1), QPointF(x, inner.bottom() - 1))
        y = round(inner.top() + inner.height() / 2.0) + 0.5
        painter.drawLine(QPointF(inner.left() + 1, y), QPointF(inner.right() - 1, y))

        painter.setFont(self._caption_font)
        painter.setPen(theme.color(theme.TEXT_DIM))
        caption_rect = inner.adjusted(6, 3, -6, -3)
        if self._hist is None:
            painter.drawText(inner, Qt.AlignCenter, "NO DATA")
            painter.drawText(caption_rect, Qt.AlignLeft | Qt.AlignTop, "HIST")
            painter.end()
            return

        # Normalise against the interior peak so full-black/full-white
        # clip spikes at bins 0/255 don't flatten everything else.
        interior_peak = float(self._hist[:, 1:255].max())
        peak = interior_peak if interior_peak > 0 else float(max(self._hist.max(), 1))

        width = inner.width() - 2
        base = inner.bottom() - 1
        height = inner.height() - 20
        for channel in range(4):
            counts = self._hist[channel]
            line = QPolygonF()
            for b in range(256):
                x = inner.left() + 1 + width * b / 255.0
                frac = min(counts[b] / peak, 1.0)
                line.append(QPointF(x, base - frac * height))
            color = theme.color(_CHANNEL_COLORS[channel])
            if channel < 3:
                fill = QPolygonF(line)
                fill.append(QPointF(inner.right() - 1, base))
                fill.append(QPointF(inner.left() + 1, base))
                fc = QColor(color)
                fc.setAlpha(46)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fc)
                painter.drawPolygon(fill)
            color.setAlpha(235 if channel == 3 else 200)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 1.5 if channel == 3 else 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPolyline(line)

        painter.setFont(self._caption_font)
        painter.setPen(theme.color(theme.TEXT_DIM))
        painter.drawText(caption_rect, Qt.AlignLeft | Qt.AlignTop, "RGB \u2022 LUMA")
        painter.end()
