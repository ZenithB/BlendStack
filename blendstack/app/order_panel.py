"""Routing panel: drag the per-image processing stages into any order.

:class:`OrderPanel` shows the six stages of
:data:`blendstack.core.adjustments.STAGES` in a drag-reorderable list; the
top row is processed first.  Each row is painted as a "chip" (structural
purple fill, white text, a grip of dots on the left).  The stage id lives in
``Qt.UserRole`` so label/id mapping never depends on the display text.  An
*active* stage (non-identity setting) is a brighter chip with a gold bar on
its left; an inactive (identity) stage is the deepest purple with italic text
(:meth:`OrderPanel.set_active_stages`).
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import DEFAULT_ORDER, STAGES

from . import theme
from .knob import ghost_button_stylesheet

__all__ = ["OrderPanel", "STAGE_LABELS", "contrast_pairs"]

#: Display names for the stage ids.
STAGE_LABELS = {
    "exposure": "Exposure",
    "contrast": "Contrast",
    "curves": "Curves",
    "saturation": "Saturation",
    "denoise": "Noise removal",
    "sharpen": "Sharpen",
}

#: Height of one chip row (px).
ROW_HEIGHT = 20

#: Item data role carrying the "stage is active" flag for the delegate.
_ACTIVE_ROLE = int(Qt.UserRole) + 1

# Chip colours: structural purple fills carry white text; an inactive
# (identity) stage is the deepest purple with the dimmer-but-readable TEXT_DIM.
_ACTIVE_FILL = (theme.PURPLE[0], theme.PURPLE[1])
_INACTIVE_FILL = theme.PURPLE[3]
_ACTIVE_TEXT = theme.TEXT_ON_PURPLE
_INACTIVE_TEXT = theme.TEXT_DIM


def contrast_pairs() -> list[tuple[str, str, str]]:
    """(name, foreground, background) pairs hard-wired in this module."""
    return [
        ("chip text (active) on purple[0]", _ACTIVE_TEXT, _ACTIVE_FILL[0]),
        ("chip text (active) on purple[1]", _ACTIVE_TEXT, _ACTIVE_FILL[1]),
        ("chip text (inactive) on purple[3]", _INACTIVE_TEXT, _INACTIVE_FILL),
        ("order hint on panel", theme.TEXT_DIM, theme.PANEL),
    ]


def _normalise(order: Iterable[str]) -> tuple[str, ...]:
    """Valid permutation of STAGES: drop unknown/duplicate ids, append missing."""
    seen: list[str] = []
    for name in order or ():
        if isinstance(name, str) and name in STAGES and name not in seen:
            seen.append(name)
    seen.extend(s for s in DEFAULT_ORDER if s not in seen)
    return tuple(seen)


class _ChipDelegate(QStyledItemDelegate):
    """Paints each stage as a purple chip: grip dots, label and, for an
    active stage, a gold bar on the left."""

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        active = bool(index.data(_ACTIVE_ROLE))
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        if not (option.state & QStyle.State_Enabled):
            painter.setOpacity(0.7)  # whole panel disabled: greyed but readable
        r = QRectF(option.rect).adjusted(1, 1, -1, -1)
        if active:
            grad = QLinearGradient(r.topLeft(), r.topRight())
            grad.setColorAt(0.0, theme.color(_ACTIVE_FILL[0]))
            grad.setColorAt(1.0, theme.color(_ACTIVE_FILL[1]))
            painter.setBrush(grad)
        else:
            painter.setBrush(theme.color(_INACTIVE_FILL))
        selected = bool(option.state & QStyle.State_Selected)
        painter.setPen(QPen(theme.color(theme.GOLD[2] if selected else theme.PURPLE[2]), 1.0))
        painter.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
        if active:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.color(theme.GOLD[0]))
            painter.drawRoundedRect(
                QRectF(r.left() + 2.0, r.top() + 3.0, 2.5, r.height() - 6.0), 1.2, 1.2
            )
        # grip: two columns of three dots
        text_col = theme.color(_ACTIVE_TEXT if active else _INACTIVE_TEXT)
        painter.setPen(Qt.NoPen)
        painter.setBrush(text_col)
        gx = r.left() + 11.0
        cy = r.center().y()
        for col in (0.0, 3.6):
            for row in (-3.4, 0.0, 3.4):
                painter.drawEllipse(QRectF(gx + col - 0.8, cy + row - 0.8, 1.6, 1.6))
        # label (uppercase, small, letter-spaced)
        font = QFont(option.font)
        font.setPixelSize(10)
        font.setBold(active)
        font.setItalic(not active)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
        painter.setFont(font)
        painter.setPen(text_col)
        text_rect = QRectF(r.left() + 24, r.top(), r.width() - 28, r.height())
        painter.drawText(
            text_rect, Qt.AlignVCenter | Qt.AlignLeft, str(index.data(Qt.DisplayRole)).upper()
        )
        painter.restore()


class _StageList(QListWidget):
    """Single-column internal-move list that sizes itself to its rows."""

    def _rows_height(self) -> int:
        n = max(1, self.count())
        return n * ROW_HEIGHT + 2 * self.frameWidth() + 2

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(150, self._rows_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(130, self._rows_height())


class OrderPanel(QWidget):
    """Drag-to-reorder list of the per-image processing stages."""

    #: Emitted after the user (or move_item / reset) changes the order.
    order_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._order: tuple[str, ...] = DEFAULT_ORDER
        self._active: set[str] = set(STAGES)

        self.title_label = QLabel("Top runs first • drag to reorder", self)
        self.title_label.setWordWrap(True)
        self.title_label.setProperty("role", "dim")
        hint_font = theme.label_font(9)
        self.title_label.setFont(hint_font)

        self.list = _StageList(self)
        self.list.setItemDelegate(_ChipDelegate(self.list))
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragEnabled(True)
        self.list.setAcceptDrops(True)
        self.list.setDropIndicatorShown(True)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.list.setFrameShape(QListWidget.NoFrame)
        self.list.setStyleSheet(
            f"QListWidget {{ background: {theme.INSET}; border: 1px solid {theme.BORDER};"
            f" border-radius: 4px; outline: none; }}"
        )
        self.list.setToolTip("Top row is applied first. Drag a row to change the order.")

        self.reset_button = QPushButton("RESET ORDER", self)
        self.reset_button.setFixedHeight(18)
        self.reset_button.setStyleSheet(ghost_button_stylesheet())
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.clicked.connect(self.reset_order)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.list)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self._rebuild(self._order)
        # A drop in an InternalMove list is a model row move.
        self.list.model().rowsMoved.connect(self._on_rows_moved)

    # ------------------------------------------------------------------- API

    def order(self) -> tuple[str, ...]:
        """The 6 stage ids in the current top-to-bottom order."""
        return self._read_list() or self._order

    def set_order(self, order: Iterable[str]) -> None:
        """Show ``order`` (normalised to a valid permutation) silently."""
        self._order = _normalise(order)
        self._rebuild(self._order)

    def set_active_stages(self, active: Iterable[str]) -> None:
        """Mark the rows whose stage id is not in ``active`` as inactive
        (no-op stages: dimmer chip, italic text)."""
        self._active = set(active)
        self._style_rows()

    def active_stages(self) -> set[str]:
        return set(self._active)

    def move_item(self, from_row: int, to_row: int) -> None:
        """Move the row at ``from_row`` so it ends up at index ``to_row``
        (list.insert semantics on the resulting list); emits order_changed."""
        n = len(self._order)
        if not (0 <= from_row < n):
            return
        to_row = max(0, min(n - 1, to_row))
        if from_row == to_row:
            return
        cur = list(self.order())
        item = cur.pop(from_row)
        cur.insert(to_row, item)
        self._order = _normalise(cur)
        self._rebuild(self._order)
        self.order_changed.emit()

    def reset_order(self) -> None:
        """Back to the default order; emits order_changed only if it changed."""
        if self.order() != DEFAULT_ORDER:
            self._order = DEFAULT_ORDER
            self._rebuild(self._order)
            self.order_changed.emit()

    def row_item(self, row: int) -> QListWidgetItem:
        return self.list.item(row)

    # -------------------------------------------------------------- internals

    def _read_list(self) -> Optional[tuple[str, ...]]:
        """The ids as currently listed, or None if the list is not a valid
        permutation of STAGES (e.g. mid-drop)."""
        ids = [
            self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
        ]
        if len(ids) == len(STAGES) and set(ids) == set(STAGES):
            return tuple(ids)
        return None

    def _rebuild(self, order: tuple[str, ...]) -> None:
        self.list.clear()
        for stage in order:
            item = QListWidgetItem(STAGE_LABELS[stage])
            item.setData(Qt.UserRole, stage)
            item.setSizeHint(QSize(0, ROW_HEIGHT))
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled
            )
            self.list.addItem(item)
        self._style_rows()
        self.list.updateGeometry()

    def _style_rows(self) -> None:
        # Italic font + an explicit foreground brush mark an inactive stage
        # (the delegate paints from these); active rows carry no brush.
        dim = theme.color(theme.TEXT_DIM)
        for i in range(self.list.count()):
            item = self.list.item(i)
            active = item.data(Qt.UserRole) in self._active
            font = item.font()
            font.setItalic(not active)
            item.setFont(font)
            item.setData(Qt.ForegroundRole, None if active else dim)
            item.setData(_ACTIVE_ROLE, active)
        self.list.viewport().update()

    def _on_rows_moved(self, *_args: object) -> None:
        got = self._read_list()
        if got is None:
            # A drop must never duplicate/lose a row: restore the last good
            # order (deferred, we are inside the model's move).
            QTimer.singleShot(0, lambda: self._rebuild(self._order))
            return
        if got != self._order:
            self._order = got
            self._style_rows()
            self.order_changed.emit()
