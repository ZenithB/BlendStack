"""Reorderable image strip (project brief §5, "Layout" left column).

A ``QListWidget`` showing, per image and left→right: a fold-order badge
(1, 2, 3 …; the first *un-muted* row is labelled BASE), the thumbnail, the
file name and two small square toggle buttons — **M** (mute) and **S**
(solo).  Top row = first = base image of the fold.  Supports:

* drag to reorder (``InternalMove``) — emits :attr:`order_changed`;
* dragging files in from Finder — emits :attr:`files_dropped`;
* per-item removal via context menu and the Delete/Backspace key —
  emits :attr:`remove_requested`;
* a per-row **Mute** toggle ("M" button, Shift+M key and a context-menu entry)
  — emits :attr:`mute_requested` (entry id, new muted state).  A muted image
  is left out of the blend; its row is dimmed with a struck-through name;
* a per-row **Solo** toggle ("S" button, the S key and a context-menu entry)
  — emits :attr:`solo_requested` (entry id, or ``None`` to un-solo);
* "Reset position" in the context menu — emits
  :attr:`reset_position_requested`.

The whole row — badge, thumbnail, name, buttons, selection / solo / muted
styling — is painted and hit-tested by a custom item delegate (NOT
``setItemWidget``, which would break InternalMove drag-reorder).  A press on a
button is swallowed, so it never starts a drag, changes the selection or alters
the fold order.

The strip never touches :class:`~blendstack.app.state.DocumentState`
directly; the main window wires the signals both ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
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

from . import theme
from .preview import array_to_qimage
from .state import ImageEntry

__all__ = ["ImageStrip"]

_ID_ROLE = Qt.UserRole
_MUTE_ROLE = Qt.UserRole + 1
_THUMB_SIZE = QSize(64, 48)

# -- row geometry (viewport pixels) ---------------------------------------------
_ROW_HEIGHT = 66
_PAD = 8                 # left/right padding
_BADGE_W = 32            # fold-order badge column
_BTN = 22                # square toggle buttons
_BTN_GAP = 4
_GAP = 8
_BAR_W = 3               # selected-row gold bar

_SOLO_TOOLTIP = "Solo – show only this image"
_MUTE_TOOLTIP = "Mute – remove from the blend"


def solo_rect_for_row(row_rect: QRect) -> QRect:
    """The Solo button rectangle inside a row's rectangle (bottom right)."""
    return QRect(
        row_rect.right() - _PAD - _BTN + 1,
        row_rect.bottom() - 8 - _BTN + 1,
        _BTN,
        _BTN,
    )


def mute_rect_for_row(row_rect: QRect) -> QRect:
    """The Mute button rectangle, immediately left of the Solo button."""
    solo = solo_rect_for_row(row_rect)
    return QRect(solo.left() - _BTN_GAP - _BTN, solo.top(), _BTN, _BTN)


def thumb_rect_for_row(row_rect: QRect) -> QRect:
    return QRect(
        row_rect.left() + _PAD + _BADGE_W + 4,
        row_rect.center().y() - _THUMB_SIZE.height() // 2,
        _THUMB_SIZE.width(),
        _THUMB_SIZE.height(),
    )


def text_rect_for_row(row_rect: QRect) -> QRect:
    """The file-name area (above the buttons, right of the thumbnail)."""
    left = thumb_rect_for_row(row_rect).right() + 1 + _GAP
    return QRect(left, row_rect.top() + 9, row_rect.right() - _PAD - left + 1, 20)


class _StripDelegate(QStyledItemDelegate):
    """Paints the whole row: badge, thumbnail, name, M / S buttons."""

    def __init__(self, strip: "ImageStrip") -> None:
        super().__init__(strip)
        self._strip = strip

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), _ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        strip = self._strip
        rect = option.rect
        entry_id = index.data(_ID_ROLE)
        muted = bool(index.data(_MUTE_ROLE))
        soloed = entry_id is not None and entry_id == strip.solo_id()
        selected = bool(option.state & QStyle.State_Selected)
        row = index.row()

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # -- background -----------------------------------------------------------
        painter.fillRect(rect, theme.color(theme.PANEL_HI if selected else theme.INSET))
        if soloed:
            painter.fillRect(rect, theme.color(theme.GOLD[0], 38))
        if option.state & QStyle.State_MouseOver and not selected:
            painter.fillRect(rect, theme.color(theme.PANEL))
        painter.setPen(theme.color(theme.BORDER, 150))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        if selected:
            painter.fillRect(
                QRect(rect.left(), rect.top(), _BAR_W, rect.height()),
                theme.color(theme.GOLD[0]),
            )

        # -- fold-order badge ----------------------------------------------------------
        self._paint_badge(painter, rect, row, muted, strip.base_row() == row)

        # -- thumbnail ------------------------------------------------------------------
        thumb = thumb_rect_for_row(rect)
        painter.fillRect(thumb, QColor(0, 0, 0))
        icon = index.data(Qt.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            pm = icon.pixmap(_THUMB_SIZE)
            target = QRect(QPoint(0, 0), pm.size())
            target.moveCenter(thumb.center())
            painter.setOpacity(0.35 if muted else 1.0)
            painter.drawPixmap(target, pm)
            painter.setOpacity(1.0)
        painter.setPen(QPen(theme.color(theme.BORDER_HI), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(thumb.adjusted(0, 0, -1, -1))

        # -- file name ---------------------------------------------------------------
        font = theme.label_font(11, bold=False)
        font.setStrikeOut(muted)
        painter.setFont(font)
        painter.setPen(theme.color(theme.TEXT_DIM if muted else theme.TEXT))
        text_rect = text_rect_for_row(rect)
        name = painter.fontMetrics().elidedText(
            index.data(Qt.DisplayRole) or "", Qt.ElideMiddle, text_rect.width()
        )
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, name)

        # -- buttons ---------------------------------------------------------------------
        self._paint_button(painter, mute_rect_for_row(rect), "M", muted, "mute")
        self._paint_button(painter, solo_rect_for_row(rect), "S", soloed, "solo")
        painter.restore()

    @staticmethod
    def _paint_badge(painter: QPainter, rect: QRect, row: int, muted: bool,
                     is_base: bool) -> None:
        box = QRect(rect.left() + _PAD, rect.top(), _BADGE_W, rect.height())
        num_font = theme.mono_font(12, bold=True)
        painter.setFont(num_font)
        painter.setPen(theme.color(theme.TEXT_DIM if muted else theme.TEXT))
        if is_base:
            painter.drawText(
                box.adjusted(0, 0, 0, -10), Qt.AlignHCenter | Qt.AlignVCenter, str(row + 1)
            )
            tag = QRect(box.left(), box.center().y() + 7, _BADGE_W, 13)
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.color(theme.GOLD[0]))
            painter.drawRoundedRect(tag, 3, 3)
            small = theme.label_font(8, bold=True)
            small.setPixelSize(10)
            painter.setFont(small)
            painter.setPen(theme.color(theme.TEXT_ON_GOLD))
            painter.drawText(tag, Qt.AlignCenter, "BASE")
        else:
            painter.drawText(box, Qt.AlignCenter, str(row + 1))

    @staticmethod
    def _paint_button(painter: QPainter, rect: QRect, letter: str, on: bool,
                      kind: str) -> None:
        box = QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
        font = theme.label_font(11, bold=True)
        painter.setFont(font)
        if on:
            fill = theme.GOLD[0] if kind == "solo" else theme.PURPLE[0]
            ink = theme.TEXT_ON_GOLD if kind == "solo" else theme.TEXT_ON_PURPLE
            painter.setPen(QPen(theme.color(
                theme.GOLD[1] if kind == "solo" else theme.PURPLE[1]), 1.0))
            painter.setBrush(theme.color(fill))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(theme.color(ink))
        else:
            painter.setPen(QPen(theme.color(theme.BORDER_HI), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(theme.color(theme.TEXT_DIM))
        painter.drawText(rect, Qt.AlignCenter, letter)

    def helpEvent(self, event, view, option, index) -> bool:  # noqa: N802
        if event.type() == QEvent.ToolTip:
            if solo_rect_for_row(option.rect).contains(event.pos()):
                QToolTip.showText(event.globalPos(), _SOLO_TOOLTIP, view)
                return True
            if mute_rect_for_row(option.rect).contains(event.pos()):
                QToolTip.showText(event.globalPos(), _MUTE_TOOLTIP, view)
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
    mute_requested = Signal(int, bool)  # entry id, desired muted state
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
        self.setWordWrap(False)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self._icons: dict[int, QIcon] = {}
        self._solo_id: Optional[int] = None
        self._button_press = False  # a press landed on an M / S button
        self.setItemDelegate(_StripDelegate(self))
        self.currentItemChanged.connect(self._on_current_changed)

    # -- state sync -----------------------------------------------------------

    def sync(self, entries: Sequence[ImageEntry]) -> None:
        """Rebuild items to mirror the document (no signals emitted)."""
        ids = [e.entry_id for e in entries]
        if ids == self.current_ids():
            for i, entry in enumerate(entries):  # membership same: flags only
                self.item(i).setData(_MUTE_ROLE, entry.adjustments.mute)
            self.viewport().update()
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
            item.setData(_MUTE_ROLE, entry.adjustments.mute)
            item.setToolTip(str(entry.path))
            self.addItem(item)
        if selected in live:
            self.setCurrentRow(ids.index(selected))
        elif entries:
            self.setCurrentRow(0)
        self.blockSignals(False)
        self.selection_changed.emit(self.current_entry_id())

    # -- mute -----------------------------------------------------------------

    def set_muted(self, entry_id: int, muted: bool) -> None:
        """Mirror one image's mute state (no signal emitted)."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(_ID_ROLE) == entry_id:
                if bool(item.data(_MUTE_ROLE)) != bool(muted):
                    item.setData(_MUTE_ROLE, bool(muted))
                    self.viewport().update()
                return

    def is_muted(self, row: int) -> bool:
        return 0 <= row < self.count() and bool(self.item(row).data(_MUTE_ROLE))

    def base_row(self) -> int:
        """Row of the first un-muted image (the fold's base), or -1."""
        for i in range(self.count()):
            if not self.item(i).data(_MUTE_ROLE):
                return i
        return -1

    def mute_button_rect(self, row: int) -> QRect:
        """Mute button rectangle of ``row`` in viewport coordinates (test /
        accessibility hook); an empty rect for an invalid row."""
        if not 0 <= row < self.count():
            return QRect()
        return mute_rect_for_row(self.visualItemRect(self.item(row)))

    def _toggle_mute(self, entry_id: Optional[int]) -> None:
        if entry_id is None:
            return
        for i in range(self.count()):
            item = self.item(i)
            if item.data(_ID_ROLE) == entry_id:
                self.mute_requested.emit(entry_id, not bool(item.data(_MUTE_ROLE)))
                return

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

    def _button_at(self, pos: QPoint) -> tuple[int, str]:
        """``(row, "mute" | "solo")`` of the button under ``pos``, or (-1, "")."""
        item = self.itemAt(pos)
        if item is None:
            return -1, ""
        row = self.row(item)
        if self.solo_button_rect(row).contains(pos):
            return row, "solo"
        if self.mute_button_rect(row).contains(pos):
            return row, "mute"
        return -1, ""

    def _toggle_solo(self, entry_id: Optional[int]) -> None:
        if entry_id is not None:
            self.solo_requested.emit(
                None if entry_id == self._solo_id else entry_id
            )

    def _press_button(self, row: int, kind: str) -> None:
        entry_id = self.item(row).data(_ID_ROLE)
        if kind == "solo":
            self._toggle_solo(entry_id)
        else:
            self._toggle_mute(entry_id)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            row, kind = self._button_at(event.position().toPoint())
            if row >= 0:
                # Swallow: no selection change, no drag, no reorder.
                self._button_press = True
                self._press_button(row, kind)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._button_press:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._button_press:
            self._button_press = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        row, kind = self._button_at(event.position().toPoint())
        if row >= 0:
            self._button_press = True  # a quick second click: toggle again
            self._press_button(row, kind)
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

    # -- keys / menu ---------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._request_remove_selected()
        elif event.key() == Qt.Key_S and not event.modifiers():
            self._toggle_solo(self.current_entry_id())
        elif event.key() == Qt.Key_M and event.modifiers() == Qt.ShiftModifier:
            self._toggle_mute(self.current_entry_id())
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if item is None:
            return
        entry_id = item.data(_ID_ROLE)
        menu = QMenu(self)
        mute = menu.addAction(
            "Unmute" if item.data(_MUTE_ROLE) else "Mute"
        )
        solo = menu.addAction(
            "Unsolo" if entry_id == self._solo_id else "Solo"
        )
        reset = menu.addAction("Reset position")
        menu.addSeparator()
        remove = menu.addAction(f"Remove “{item.text()}”")
        chosen = menu.exec(event.globalPos())
        if chosen is mute:
            self._toggle_mute(entry_id)
        elif chosen is solo:
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

    # -- painting (empty hint) -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.count() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(theme.color(theme.TEXT_DIM))
            painter.setFont(theme.label_font(12))
            painter.drawText(
                self.viewport().rect().adjusted(14, 0, -14, 0),
                Qt.AlignCenter | Qt.TextWordWrap,
                "Drop images here\nor press Open",
            )
            painter.end()
