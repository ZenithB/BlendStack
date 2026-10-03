"""Routing panel: drag the per-image processing stages into any order.

:class:`OrderPanel` shows the six stages of
:data:`blendstack.core.adjustments.STAGES` in a drag-reorderable list; the
top row is processed first.  Each row's text carries a leading drag-handle
glyph, and the stage id lives in ``Qt.UserRole`` so label/id mapping never
depends on the display text.  Stages that are currently a no-op (identity
setting) are drawn dimmed and italic (:meth:`OrderPanel.set_active_stages`).
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import DEFAULT_ORDER, STAGES

__all__ = ["OrderPanel", "STAGE_LABELS"]

#: Display names for the stage ids.
STAGE_LABELS = {
    "exposure": "Exposure",
    "contrast": "Contrast",
    "curves": "Curves",
    "saturation": "Saturation",
    "denoise": "Noise removal",
    "sharpen": "Sharpen",
}

_HANDLE = "⋮⋮  "  # "⋮⋮  " drag-handle glyph


def _normalise(order: Iterable[str]) -> tuple[str, ...]:
    """Valid permutation of STAGES: drop unknown/duplicate ids, append missing."""
    seen: list[str] = []
    for name in order or ():
        if isinstance(name, str) and name in STAGES and name not in seen:
            seen.append(name)
    seen.extend(s for s in DEFAULT_ORDER if s not in seen)
    return tuple(seen)


class _StageList(QListWidget):
    """Single-column internal-move list that sizes itself to its rows."""

    def _rows_height(self) -> int:
        n = max(1, self.count())
        row = self.sizeHintForRow(0) if self.count() else 18
        return n * max(row, 16) + 2 * self.frameWidth() + 2

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, self._rows_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(120, self._rows_height())


class OrderPanel(QWidget):
    """Drag-to-reorder list of the per-image processing stages."""

    #: Emitted after the user (or move_item / reset) changes the order.
    order_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._order: tuple[str, ...] = DEFAULT_ORDER
        self._active: set[str] = set(STAGES)

        self.title_label = QLabel("Processing order — drag to reorder", self)
        self.title_label.setWordWrap(True)

        self.list = _StageList(self)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragEnabled(True)
        self.list.setAcceptDrops(True)
        self.list.setDropIndicatorShown(True)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.list.setAlternatingRowColors(True)
        self.list.setToolTip("Top row is applied first. Drag a row to change the order.")

        self.reset_button = QPushButton("Reset order", self)
        self.reset_button.clicked.connect(self.reset_order)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.list)
        layout.addWidget(self.reset_button)

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
        """Dim the rows whose stage id is not in ``active`` (no-op stages)."""
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
            item = QListWidgetItem(_HANDLE + STAGE_LABELS[stage])
            item.setData(Qt.UserRole, stage)
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled
            )
            self.list.addItem(item)
        self._style_rows()
        self.list.updateGeometry()

    def _style_rows(self) -> None:
        dim = self.palette().placeholderText()
        for i in range(self.list.count()):
            item = self.list.item(i)
            active = item.data(Qt.UserRole) in self._active
            font = item.font()
            font.setItalic(not active)
            item.setFont(font)
            item.setData(Qt.ForegroundRole, None if active else dim)

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
