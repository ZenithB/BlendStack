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
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QPointF

__all__ = ["HistogramWidget"]

#: Draw order: R, G, B first, luma on top.
_CHANNEL_COLORS = (
    QColor(235, 90, 90),    # R
    QColor(110, 210, 110),  # G
    QColor(110, 140, 245),  # B
    QColor(235, 235, 235),  # Rec.709 luma
)


class HistogramWidget(QWidget):
    """Paints a (4, 256) bin-count array as four overlaid curves."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hist: Optional[np.ndarray] = None
        self.setMinimumHeight(110)

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
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QColor(24, 24, 26))
        painter.setPen(QColor(70, 70, 74))
        painter.drawRect(rect)
        for i in (1, 2, 3):  # quarter gridlines
            x = rect.left() + rect.width() * i / 4.0
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        if self._hist is None:
            painter.setPen(QColor(120, 120, 125))
            painter.drawText(rect, Qt.AlignCenter, "Histogram")
            painter.end()
            return

        # Normalise against the interior peak so full-black/full-white
        # clip spikes at bins 0/255 don't flatten everything else.
        interior_peak = float(self._hist[:, 1:255].max())
        peak = interior_peak if interior_peak > 0 else float(max(self._hist.max(), 1))

        width = rect.width()
        height = rect.height() - 2
        for channel in range(4):
            counts = self._hist[channel]
            points = QPolygonF()
            for b in range(256):
                x = rect.left() + width * b / 255.0
                frac = min(counts[b] / peak, 1.0)
                y = rect.bottom() - 1 - frac * height
                points.append(QPointF(x, y))
            color = QColor(_CHANNEL_COLORS[channel])
            color.setAlpha(230 if channel == 3 else 190)
            painter.setPen(QPen(color, 1.2))
            painter.drawPolyline(points)
        painter.end()
