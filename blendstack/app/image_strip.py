"""Reorderable image strip (project brief §5, "Layout" left column).

A ``QListWidget`` showing thumbnail + filename per image.  Top item = first
= base image of the fold.  Supports:

* drag to reorder (``InternalMove``) — emits :attr:`order_changed`;
* dragging files in from Finder — emits :attr:`files_dropped`;
* per-item removal via context menu and the Delete/Backspace key —
  emits :attr:`remove_requested`;
* a per-row **Solo** toggle ("S" button at the row's right edge, the S key
  and a context-menu entry) — emits :attr:`solo_requested` (entry id, or
  ``None`` to un-solo).  The button is painted and hit-tested by a custom
  item delegate (NOT ``setItemWidget``, which would break InternalMove
  drag-reorder).  A press on the button is swallowed, so it never starts a
  drag, changes the selection or alters the fold order;
* "Reset position" in the context menu — emits
  :attr:`reset_position_requested`.

The strip never touches :class:`~blendstack.app.state.DocumentState`
directly; the main window wires the signals both ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QToolTip,
    QWidget,
)

from blendstack.core import io as bs_io

from .preview import array_to_qimage
from .state import ImageEntry

__all__ = ["ImageStrip"]

_ID_ROLE = Qt.UserRole
_THUMB_SIZE = QSize(96, 64)

#: Solo button geometry (viewport pixels, at the row's right edge).
_SOLO_SIZE = 22
_SOLO_MARGIN = 6
_SOLO_COLUMN = _SOLO_SIZE + 2 * _SOLO_MARGIN  # text never runs under it
_SOLO_TOOLTIP = "Solo – show only this image"
_SOLO_ACCENT = QColor(255, 170, 0)


def solo_rect_for_row(row_rect: QRect) -> QRect:
    """The Solo button rectangle inside a row's rectangle."""
    return QRect(
        row_rect.right() - _SOLO_MARGIN - _SOLO_SIZE + 1,
        row_rect.center().y() - _SOLO_SIZE // 2,
        _SOLO_SIZE,
        _SOLO_SIZE,
    )


class _StripDelegate(QStyledItemDelegate):
    """Paints the usual row plus the Solo button / soloed-row highlight."""

    def __init__(self, strip: "ImageStrip") -> None:
        super().__init__(strip)
        self._strip = strip

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        entry_id = index.data(_ID_ROLE)
        soloed = entry_id is not None and entry_id == self._strip.solo_id()
        painter.save()
        if soloed:  # highlight tint behind the row
            tint = QColor(_SOLO_ACCENT)
            tint.setAlpha(55)
            painter.fillRect(option.rect, tint)
        painter.restore()

        # Let the style draw the row in the area left of the button column.
        narrowed = type(option)(option)
        content = QRect(option.rect)
        content.setRight(option.rect.right() - _SOLO_COLUMN)
        narrowed.rect = content
        super().paint(painter, narrowed, index)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if option.state & QStyle.State_Selected:  # extend the selection fill
            group = (
                QPalette.Active if option.state & QStyle.State_Active
                else QPalette.Inactive
            )
            painter.fillRect(
                QRect(narrowed.rect.right() + 1, option.rect.top(),
                      _SOLO_COLUMN, option.rect.height()),
                option.palette.brush(group, QPalette.Highlight),
            )
            if soloed:
                tint = QColor(_SOLO_ACCENT)
                tint.setAlpha(55)
                painter.fillRect(option.rect, tint)
        if soloed:  # accent bar on the row's left edge
            painter.fillRect(
                QRect(option.rect.left(), option.rect.top(), 3,
                      option.rect.height()),
                _SOLO_ACCENT,
            )
        self._paint_button(painter, solo_rect_for_row(option.rect), soloed, option)
        painter.restore()

    @staticmethod
    def _paint_button(painter: QPainter, rect: QRect, on: bool, option) -> None:
        box = rect.adjusted(1, 1, -1, -1)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSizeF(max(font.pointSizeF() - 1.0, 8.0))
        painter.setFont(font)
        if on:
            painter.setPen(Qt.NoPen)
            painter.setBrush(_SOLO_ACCENT)
            painter.drawRoundedRect(box, 5, 5)
            painter.setPen(QColor(30, 20, 0))
        else:
            dim = QColor(option.palette.color(QPalette.Text))
            dim.setAlpha(120)
            painter.setPen(QPen(dim, 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box, 5, 5)
        painter.drawText(box, Qt.AlignCenter, "S")

    def helpEvent(self, event, view, option, index) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.ToolTip
            and solo_rect_for_row(option.rect).contains(event.pos())
        ):
            QToolTip.showText(event.globalPos(), _SOLO_TOOLTIP, view)
            return True
        return super().helpEvent(event, view, option, index)


def _thumbnail_icon(entry: ImageEntry) -> QIcon:
    """Build a strip thumbnail from the entry's preview proxy."""
    proxy = entry.proxy
    # Cheap pre-decimation so the smooth scale below stays fast.
    step = max(1, max(proxy.shape[:2]) // 256)
    small = proxy[::step, ::step]
    pixmap = QPixmap.fromImage(array_to_qimage(small)).scaled(
        _THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    return QIcon(pixmap)


def urls_to_supported_paths(urls: Iterable) -> list[Path]:
    """Local-file URLs → paths, keeping any supported-extension file."""
    paths: list[Path] = []
    for url in urls:
        if url.isLocalFile():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in bs_io.SUPPORTED_INPUT_EXTENSIONS:
                paths.append(path)
    return paths


class ImageStrip(QListWidget):
    """Thumbnail list; order = fold order (top = first/base image)."""

    files_dropped = Signal(list)     # list[Path] dropped from Finder
    order_changed = Signal(list)     # list[int] entry ids, new order
    remove_requested = Signal(list)  # list[int] entry ids
    selection_changed = Signal(object)  # entry id (int) or None
    solo_requested = Signal(object)  # entry id to solo, or None to un-solo
    reset_position_requested = Signal(int)  # entry id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setIconSize(_THUMB_SIZE)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setUniformItemSizes(True)
        self.setWordWrap(True)
        self.setMinimumWidth(190)
        self.setMaximumWidth(280)
        self._icons: dict[int, QIcon] = {}
        self._solo_id: Optional[int] = None
        self._solo_press = False  # a press landed on a Solo button
        self.setItemDelegate(_StripDelegate(self))
        self.currentItemChanged.connect(self._on_current_changed)

    # -- state sync -----------------------------------------------------------

    def sync(self, entries: Sequence[ImageEntry]) -> None:
        """Rebuild items to mirror the document (no signals emitted)."""
        ids = [e.entry_id for e in entries]
        if ids == self.current_ids():
            return  # order and membership unchanged
        selected = self.current_entry_id()
        self.blockSignals(True)
        self.clear()
        live = set(ids)
        for stale in [i for i in self._icons if i not in live]:
            del self._icons[stale]
        for entry in entries:
            icon = self._icons.get(entry.entry_id)
            if icon is None:
                icon = _thumbnail_icon(entry)
                self._icons[entry.entry_id] = icon
            item = QListWidgetItem(icon, entry.path.name)
            item.setData(_ID_ROLE, entry.entry_id)
            item.setToolTip(str(entry.path))
            self.addItem(item)
        if selected in live:
            self.setCurrentRow(ids.index(selected))
        elif entries:
            self.setCurrentRow(0)
        self.blockSignals(False)
        self.selection_changed.emit(self.current_entry_id())

    # -- solo -----------------------------------------------------------------

    def solo_id(self) -> Optional[int]:
        """The entry id shown as soloed (mirrors the document state)."""
        return self._solo_id

    def set_solo_id(self, entry_id: Optional[int]) -> None:
        """Mirror the document's solo state (no signal emitted)."""
        if entry_id != self._solo_id:
            self._solo_id = entry_id
            self.viewport().update()

    def solo_button_rect(self, row: int) -> QRect:
        """Solo button rectangle of ``row`` in viewport coordinates (test /
        accessibility hook); an empty rect for an invalid row."""
        if not 0 <= row < self.count():
            return QRect()
        return solo_rect_for_row(self.visualItemRect(self.item(row)))

    def _solo_row_at(self, pos: QPoint) -> int:
        """Row whose Solo button contains ``pos``, or -1."""
        item = self.itemAt(pos)
        if item is None:
            return -1
        row = self.row(item)
        return row if self.solo_button_rect(row).contains(pos) else -1

    def _toggle_solo(self, entry_id: Optional[int]) -> None:
        if entry_id is not None:
            self.solo_requested.emit(
                None if entry_id == self._solo_id else entry_id
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            row = self._solo_row_at(event.position().toPoint())
            if row >= 0:
                # Swallow: no selection change, no drag, no reorder.
                self._solo_press = True
                self._toggle_solo(self.item(row).data(_ID_ROLE))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._solo_press:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._solo_press:
            self._solo_press = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        row = self._solo_row_at(event.position().toPoint())
        if row >= 0:
            self._solo_press = True  # a quick second click: toggle again
            self._toggle_solo(self.item(row).data(_ID_ROLE))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def current_ids(self) -> list[int]:
        return [self.item(i).data(_ID_ROLE) for i in range(self.count())]

    def current_entry_id(self) -> Optional[int]:
        item = self.currentItem()
        return None if item is None else item.data(_ID_ROLE)

    def move_item(self, from_row: int, to_row: int) -> None:
        """Programmatic reorder (used by the self-test; mirrors a drag)."""
        item = self.takeItem(from_row)
        self.insertItem(to_row, item)
        self.setCurrentItem(item)
        self.order_changed.emit(self.current_ids())

    # -- drag and drop -----------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.source() is self:
            super().dropEvent(event)  # internal reorder
            self.order_changed.emit(self.current_ids())
        elif event.mimeData().hasUrls():
            paths = urls_to_supported_paths(event.mimeData().urls())
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # -- removal ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._request_remove_selected()
        elif event.key() == Qt.Key_S and not event.modifiers():
            self._toggle_solo(self.current_entry_id())
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is None:
            return
        entry_id = item.data(_ID_ROLE)
        menu = QMenu(self)
        solo = menu.addAction(
            "Unsolo" if entry_id == self._solo_id else "Solo"
        )
        reset = menu.addAction("Reset position")
        menu.addSeparator()
        remove = menu.addAction(f"Remove “{item.text()}”")
        chosen = menu.exec(event.globalPos())
        if chosen is solo:
            self._toggle_solo(entry_id)
        elif chosen is reset:
            self.reset_position_requested.emit(entry_id)
        elif chosen is remove:
            self.remove_requested.emit([entry_id])

    def _request_remove_selected(self) -> None:
        ids = [item.data(_ID_ROLE) for item in self.selectedItems()]
        if ids:
            self.remove_requested.emit(ids)

    def _on_current_changed(self, current, _previous) -> None:
        self.selection_changed.emit(
            None if current is None else current.data(_ID_ROLE)
        )
