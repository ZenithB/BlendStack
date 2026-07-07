"""Reorderable image strip (project brief §5, "Layout" left column).

A ``QListWidget`` showing thumbnail + filename per image.  Top item = first
= base image of the fold.  Supports:

* drag to reorder (``InternalMove``) — emits :attr:`order_changed`;
* dragging files in from Finder — emits :attr:`files_dropped`;
* per-item removal via context menu and the Delete/Backspace key —
  emits :attr:`remove_requested`.

The strip never touches :class:`~blendstack.app.state.DocumentState`
directly; the main window wires the signals both ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QWidget,
)

from blendstack.core import io as bs_io

from .preview import array_to_qimage
from .state import ImageEntry

__all__ = ["ImageStrip"]

_ID_ROLE = Qt.UserRole
_THUMB_SIZE = QSize(96, 64)


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
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is None:
            return
        menu = QMenu(self)
        remove = menu.addAction(f"Remove “{item.text()}”")
        if menu.exec(event.globalPos()) is remove:
            self.remove_requested.emit([item.data(_ID_ROLE)])

    def _request_remove_selected(self) -> None:
        ids = [item.data(_ID_ROLE) for item in self.selectedItems()]
        if ids:
            self.remove_requested.emit(ids)

    def _on_current_changed(self, current, _previous) -> None:
        self.selection_changed.emit(
            None if current is None else current.data(_ID_ROLE)
        )
