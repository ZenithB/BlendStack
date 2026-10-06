"""Main window — assembles the standalone app (project brief §5).

A one-screen "synth rack" layout, nothing scrolls::

    ┌ toolbar: logo · Open · Save · Load · [Move/Crop LED] ……………… EXPORT ┐
    │ image strip (≈240 px)  │  preview canvas (expands; crop bar under it) │
    ├ rack (full width): BlendControls │ AdjustmentsPanel │ HistogramWidget ┤
    └ status bar ────────────────────────────────────────────────────────┘

The strip (left) is the reorderable fold list — top = base image — with a
per-row **M**ute and **S**olo button.  A muted image is removed from the blend
(it still sizes the canvas, so nothing shifts when it is toggled; Shift+M on
the selected row).  The bottom rack holds the global blend knobs, the
per-image adjustments (IMAGE knobs | CURVES | ORDER) and the composite
histogram; it takes its height from those panels.

Toolbar: Open, Save Preset, Load Preset, the checkable **Move / Crop** tool
(key M, shown as an LED toggle) and, on the right, a gold **Export** button.

Move / Crop tool (on): the preview shows the FULL canvas; dragging in the
preview moves the selected image (``move_x`` / ``move_y`` fractions, stored in
its :class:`Adjustments`); double-clicking enters crop mode (rectangle with 8
handles; Apply / Cancel / Reset crop in a slim bar under the canvas, Enter /
Esc as shortcuts).  Solo previews one image in isolation (overrides mute).
Export honours placement, crop and mute.

Export (brief §5 / §4.4) runs the **full-resolution** pipeline via
``engine.blend_files`` (streams one file at a time, memory-bounded) on a
background thread with a modal indeterminate progress dialog — the UI
thread is never blocked.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QPen,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from blendstack import __version__
from blendstack.core import engine, geometry
from blendstack.core import io as bs_io
from blendstack.core.adjustments import Adjustments

from . import canvas_tools, presets, theme
from .adjustments_panel import AdjustmentsPanel
from .blend_controls import BlendControls
from .histogram import HistogramWidget
from .image_strip import ImageStrip, urls_to_supported_paths
from .preview import PreviewCanvas, PreviewController
from .state import AddReport, DocumentState

__all__ = ["MainWindow"]

_OPEN_FILTER = "Images ({})".format(
    " ".join(f"*{ext}" for ext in sorted(bs_io.SUPPORTED_INPUT_EXTENSIONS))
)

#: Export dialog filters (brief §4.4) and their engine format names.
_EXPORT_FILTERS = (
    ("TIFF, 16-bit (*.tif *.tiff)", "tiff"),
    ("PNG, 16-bit (*.png)", "png"),
    ("JPEG, 8-bit (*.jpg *.jpeg)", "jpeg"),
)

#: Width of the image-strip column.
_STRIP_WIDTH = 244


class _ExportWorker(QObject):
    """Runs ``engine.blend_files`` off the UI thread (brief §5 export)."""

    finished = Signal(object)  # Path of the written file
    failed = Signal(str)

    def __init__(
        self,
        paths: Sequence[Path],
        mode: str,
        params: dict[str, Any],
        adjustments: Sequence[Adjustments],
        out_path: Path,
        out_format: str,
        crop: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self._paths = list(paths)
        self._mode = mode
        self._params = dict(params)
        self._adjustments = list(adjustments)
        self._out_path = out_path
        self._out_format = out_format
        self._crop = None if crop is None else tuple(crop)

    @Slot()
    def run(self) -> None:
        try:
            written = engine.blend_files(
                self._paths,
                mode=self._mode,
                params=self._params,
                adjustments=self._adjustments,
                out_path=self._out_path,
                out_format=self._out_format,
                crop=self._crop,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced in a dialog
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(written)


# ----------------------------------------------------------------- shell widgets


class _Logo(QWidget):
    """"BLEND" (text colour) + "STACK" (gold), bold and letter-spaced, with a
    tiny dim monospace version tag."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._font = theme.label_font(15, bold=True)
        self._font.setLetterSpacing(QFont.AbsoluteSpacing, 3.0)
        self._tag_font = theme.mono_font(9)
        self._tag = f"v{__version__}"
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _widths(self) -> tuple[int, int, int]:
        fm = QFontMetrics(self._font)
        return (
            fm.horizontalAdvance("BLEND"),
            fm.horizontalAdvance("STACK"),
            QFontMetrics(self._tag_font).horizontalAdvance(self._tag),
        )

    def sizeHint(self) -> QSize:  # noqa: N802
        blend, stack, tag = self._widths()
        return QSize(blend + stack + tag + 18, 30)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        blend, stack, _tag = self._widths()
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        base = (self.height() + fm.ascent() - fm.descent()) // 2
        p.setPen(theme.color(theme.TEXT))
        p.drawText(0, base, "BLEND")
        p.setPen(theme.color(theme.GOLD[0]))
        p.drawText(blend, base, "STACK")
        p.setFont(self._tag_font)
        p.setPen(theme.color(theme.TEXT_DIM))
        p.drawText(blend + stack + 10, base, self._tag)
        p.end()


class _LedToolButton(QToolButton):
    """A checkable tool button with an LED dot: dim when off, glowing green
    when on.  Wraps a ``QAction`` (the action stays the public API)."""

    _STYLE = f"""
    QToolButton {{ background: {theme.INSET}; color: {theme.TEXT};
        border: 1px solid {theme.BORDER_HI}; border-radius: 4px;
        padding: 5px 12px 5px 30px; }}
    QToolButton:hover {{ border-color: {theme.GOLD[0]}; color: {theme.TEXT}; }}
    QToolButton:checked {{ background: {theme.INSET}; color: {theme.TEXT};
        border-color: {theme.GREEN[0]}; }}
    """

    def __init__(self, action: QAction, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setDefaultAction(action)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setStyleSheet(self._STYLE)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        c = QPointF(16.0, self.height() / 2.0)
        if self.isChecked():
            glow = QRadialGradient(c, 9.0)
            glow.setColorAt(0.0, theme.color(theme.GREEN[0], 150))
            glow.setColorAt(1.0, theme.color(theme.GREEN[0], 0))
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawEllipse(c, 9.0, 9.0)
            p.setBrush(theme.color(theme.GREEN[0]))
            p.setPen(QPen(theme.color(theme.GREEN[3]), 1.0))
        else:
            p.setBrush(theme.color(theme.GREEN[4]))
            p.setPen(QPen(theme.color(theme.BORDER_HI), 1.0))
        p.drawEllipse(c, 4.5, 4.5)
        p.end()


class _Rack(QFrame):
    """The bottom rack: a slightly raised panel spanning the window with a
    thin outline, a top highlight and gold rack screws in
    the "ears" at either end."""

    EAR = 16  # width of the screw ears left and right

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("rack")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()
        p.fillRect(r, theme.color(theme.PANEL))
        # top border + a subtle highlight line just under it
        p.setPen(QPen(theme.color(theme.BORDER), 1.0))
        p.drawLine(0, 0, r.width(), 0)
        p.setPen(QPen(theme.color(theme.PANEL_HI), 1.0))
        p.drawLine(0, 1, r.width(), 1)
        # ears (rails) with screws
        rail = theme.color(theme.BG, 120)
        p.fillRect(0, 2, self.EAR, r.height() - 2, rail)
        p.fillRect(r.width() - self.EAR, 2, self.EAR, r.height() - 2, rail)
        p.setPen(QPen(theme.color(theme.BORDER), 1.0))
        p.drawLine(self.EAR, 2, self.EAR, r.height())
        p.drawLine(r.width() - self.EAR, 2, r.width() - self.EAR, r.height())
        for x in (self.EAR / 2.0, r.width() - self.EAR / 2.0):
            for y in (14.0, r.height() - 14.0):
                self._screw(p, QPointF(x, y))
        p.end()

    @staticmethod
    def _screw(p: QPainter, c: QPointF) -> None:
        grad = QRadialGradient(c - QPointF(1.0, 1.0), 6.0)
        grad.setColorAt(0.0, theme.color(theme.GOLD[0]))
        grad.setColorAt(1.0, theme.color(theme.GOLD[3]))
        p.setPen(QPen(theme.color(theme.GOLD[4]), 1.0))
        p.setBrush(grad)
        p.drawEllipse(c, 4.0, 4.0)
        p.setPen(QPen(theme.color(theme.GOLD[4]), 1.2))
        p.drawLine(QPointF(c.x() - 2.4, c.y() + 0.6), QPointF(c.x() + 2.4, c.y() - 0.6))


class MainWindow(QMainWindow):
    """The BlendStack standalone app window (brief §5)."""

    #: (success, message) — emitted when a background export completes.
    export_done = Signal(bool, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        theme.apply(QApplication.instance())
        self.setWindowTitle("BlendStack")
        self.setAcceptDrops(True)

        self.state = DocumentState(self)
        #: Last user-facing notice as (title, text) — inspected by selftest.
        self.last_notice: Optional[tuple[str, str]] = None
        self.last_export_path: Optional[Path] = None
        self._last_dir = Path.home()
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[_ExportWorker] = None
        self._export_progress: Optional[QProgressDialog] = None
        #: (entry_id, move_x, move_y) captured when a layer drag starts.
        self._move_base: Optional[tuple[int, float, float]] = None

        # -- widgets ---------------------------------------------------------
        self.strip = ImageStrip(self)
        self.canvas = PreviewCanvas(self)
        self.adjustments_panel = AdjustmentsPanel(self)
        self.blend_controls = BlendControls(self)
        self.histogram = HistogramWidget(self)

        # Left column: the image strip under a small header.
        strip_title = QLabel("IMAGES", self)
        strip_title.setProperty("role", "title")
        strip_hint = QLabel("fold order · top = base", self)
        strip_hint.setProperty("role", "dim")
        strip_hint.setFont(theme.label_font(10))
        strip_header = QHBoxLayout()
        strip_header.setContentsMargins(2, 0, 2, 0)
        strip_header.addWidget(strip_title)
        strip_header.addStretch(1)
        strip_header.addWidget(strip_hint)
        strip_column = QVBoxLayout()
        strip_column.setContentsMargins(0, 0, 0, 0)
        strip_column.setSpacing(6)
        strip_column.addLayout(strip_header)
        strip_column.addWidget(self.strip, 1)
        strip_holder = QWidget(self)
        strip_holder.setLayout(strip_column)
        strip_holder.setFixedWidth(_STRIP_WIDTH)

        # Slim crop bar attached under the canvas; visible only in crop mode.
        self.crop_bar = QFrame(self)
        self.crop_bar.setProperty("panel", True)
        crop_bar_layout = QHBoxLayout(self.crop_bar)
        crop_bar_layout.setContentsMargins(10, 4, 8, 4)
        crop_title = QLabel("CROP", self.crop_bar)
        crop_title.setProperty("role", "title")
        crop_hint = QLabel("drag the handles or the box · Enter = Apply · Esc = Cancel",
                           self.crop_bar)
        crop_hint.setProperty("role", "dim")
        crop_hint.setFont(theme.label_font(10))
        crop_bar_layout.addWidget(crop_title)
        crop_bar_layout.addSpacing(8)
        crop_bar_layout.addWidget(crop_hint)
        crop_bar_layout.addStretch(1)
        self.reset_crop_button = QPushButton("Reset crop", self.crop_bar)
        self.cancel_crop_button = QPushButton("Cancel", self.crop_bar)
        self.apply_crop_button = QPushButton("Apply", self.crop_bar)
        self.reset_crop_button.setToolTip("Remove the applied crop")
        self.cancel_crop_button.setToolTip("Discard this crop edit (Esc)")
        self.apply_crop_button.setToolTip("Apply the crop (Enter)")
        self.reset_crop_button.setProperty("variant", "ghost")
        self.cancel_crop_button.setProperty("variant", "ghost")
        self.apply_crop_button.setProperty("variant", "primary")
        self.apply_crop_button.setDefault(True)
        for button in (self.reset_crop_button, self.cancel_crop_button,
                       self.apply_crop_button):
            button.setFixedHeight(26)
            crop_bar_layout.addWidget(button)
        self.crop_bar.hide()

        # Header above the canvas mirrors the strip's header so both columns
        # start on the same line; the readout shows the frame being displayed.
        preview_title = QLabel("PREVIEW", self)
        preview_title.setProperty("role", "title")
        self.preview_info = QLabel("", self)
        self.preview_info.setProperty("role", "dim")
        self.preview_info.setFont(theme.mono_font(10))
        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(2, 0, 2, 0)
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_info)

        canvas_column = QVBoxLayout()
        canvas_column.setContentsMargins(0, 0, 0, 0)
        canvas_column.setSpacing(6)
        canvas_column.addLayout(preview_header)
        canvas_column.addWidget(self.canvas, 1)
        canvas_column.addWidget(self.crop_bar)

        top = QWidget(self)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(12, 12, 12, 10)
        top_layout.setSpacing(12)
        top_layout.addWidget(strip_holder)
        top_layout.addLayout(canvas_column, 1)

        # Bottom rack: blend controls | adjustments (the slack) | histogram.
        self.rack = _Rack(self)
        rack_layout = QHBoxLayout(self.rack)
        rack_layout.setContentsMargins(_Rack.EAR + 12, 6, _Rack.EAR + 12, 6)
        rack_layout.setSpacing(16)
        rack_layout.addWidget(self.blend_controls, 0)
        rack_layout.addWidget(self.adjustments_panel, 1)
        rack_layout.addWidget(self.histogram, 0)

        central = QWidget(self)
        central.setObjectName("central")
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(top, 1)
        central_layout.addWidget(self.rack, 0)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._build_status_bar()
        # Enter = Apply / Esc = Cancel, active only while cropping.
        self._crop_shortcuts = [
            QShortcut(QKeySequence(key), self, self._apply_crop)
            for key in (Qt.Key_Return, Qt.Key_Enter)
        ] + [QShortcut(QKeySequence(Qt.Key_Escape), self, self._cancel_crop)]
        for shortcut in self._crop_shortcuts:
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setEnabled(False)

        # -- preview controller (background render thread, brief §5) ---------
        self.preview = PreviewController(self.state, self)
        self.preview.preview_ready.connect(self._on_preview_ready)
        self.preview.preview_cleared.connect(self._on_preview_cleared)
        self.preview.render_failed.connect(
            lambda msg: self._notify("Preview error", msg)
        )

        # -- wiring: state -> UI ----------------------------------------------
        self.state.images_changed.connect(self._sync_strip)
        self.state.blend_changed.connect(
            lambda: self.blend_controls.set_from_state(
                self.state.mode, self.state.params
            )
        )
        self.state.adjustments_changed.connect(self._on_state_adjustments)
        self.state.solo_changed.connect(self._on_solo_changed)
        self.state.crop_changed.connect(self._on_crop_changed)

        # -- wiring: UI -> state ------------------------------------------------
        self.strip.files_dropped.connect(self.add_files)
        self.strip.order_changed.connect(self.state.reorder)
        self.strip.remove_requested.connect(self.state.remove)
        self.strip.selection_changed.connect(self._on_selection_changed)
        self.strip.solo_requested.connect(self.state.set_solo)
        self.strip.mute_requested.connect(self._on_mute_requested)
        self.strip.reset_position_requested.connect(self.state.reset_placement)
        self.adjustments_panel.adjustments_edited.connect(self._on_panel_edited)
        self.blend_controls.mode_changed.connect(self.state.set_mode)
        self.blend_controls.param_changed.connect(self.state.set_param)

        # -- wiring: Move / Crop tool -------------------------------------------
        self.move_crop_action.toggled.connect(self._on_tool_toggled)
        self.canvas.drag_started.connect(self._on_move_started)
        self.canvas.drag_moved.connect(self._on_move_dragged)
        self.canvas.drag_finished.connect(self._on_move_finished)
        self.canvas.crop_mode_changed.connect(self._on_crop_mode_changed)
        self.apply_crop_button.clicked.connect(self._apply_crop)
        self.cancel_crop_button.clicked.connect(self._cancel_crop)
        self.reset_crop_button.clicked.connect(self._reset_crop)

        self.blend_controls.set_from_state(self.state.mode, self.state.params)
        self.canvas.clear()
        self._update_info()

        # The layout's true minimum is the window minimum; size/centre on the
        # screen the window opens on (still freely resizable and movable).
        self.setMinimumSize(self.minimumSizeHint())
        self._apply_initial_geometry()

    # ------------------------------------------------------------------ toolbar

    def _apply_initial_geometry(self) -> None:
        """Open at a comfortable size that always fits the screen, centred.

        The window stays freely movable and resizable afterwards; its minimum
        is the layout's true minimum size.
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:  # no screen info (rare): keep a sane default
            self.resize(1280, 800)
            return
        avail = screen.availableGeometry()
        min_hint = self.minimumSizeHint()
        width = max(min(1280, int(avail.width() * 0.92)), min_hint.width())
        height = max(min(800, int(avail.height() * 0.88)), min_hint.height())
        self.resize(width, height)
        self.move(
            avail.x() + max(0, (avail.width() - width) // 2),
            avail.y() + max(0, (avail.height() - height) // 2),
        )

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        self.toolbar = toolbar

        self.open_action = QAction("Open…", self)
        self.open_action.triggered.connect(self._open_dialog)
        self.save_preset_action = QAction("Save Preset…", self)
        self.save_preset_action.triggered.connect(self._save_preset_dialog)
        self.load_preset_action = QAction("Load Preset…", self)
        self.load_preset_action.triggered.connect(self._load_preset_dialog)
        self.export_action = QAction("Export…", self)
        self.export_action.triggered.connect(self._export_dialog)
        self.move_crop_action = QAction("Move / Crop", self)
        self.move_crop_action.setCheckable(True)
        self.move_crop_action.setShortcut(QKeySequence("M"))
        self.move_crop_action.setToolTip(
            "Move / Crop (M) — drag to move the selected image; "
            "double-click the canvas to crop"
        )
        # Shift+M: mute / un-mute the image selected in the strip (plain M stays
        # the Move / Crop tool).
        self.mute_action = QAction("Mute / Unmute selected image", self)
        self.mute_action.setShortcut(QKeySequence("Shift+M"))
        self.mute_action.setShortcutContext(Qt.WindowShortcut)
        self.mute_action.triggered.connect(self._toggle_selected_mute)
        self.addAction(self.mute_action)

        toolbar.addWidget(_Logo(toolbar))
        spacer = QWidget(toolbar)
        spacer.setFixedWidth(14)
        toolbar.addWidget(spacer)
        for action in (self.open_action, self.save_preset_action,
                       self.load_preset_action):
            toolbar.addAction(action)
            button = toolbar.widgetForAction(action)
            if button is not None:
                button.setProperty("variant", "ghost")
                button.setCursor(Qt.PointingHandCursor)
        toolbar.addSeparator()
        self.move_crop_button = _LedToolButton(self.move_crop_action, toolbar)
        toolbar.addWidget(self.move_crop_button)

        stretch = QWidget(toolbar)
        stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(stretch)

        self.export_button = QPushButton("EXPORT", toolbar)
        self.export_button.setProperty("variant", "primary")
        self.export_button.setToolTip("Export the full-resolution blend")
        self.export_button.setCursor(Qt.PointingHandCursor)
        self.export_button.setMinimumWidth(110)
        self.export_button.setFixedHeight(30)
        self.export_button.clicked.connect(self.export_action.trigger)
        toolbar.addWidget(self.export_button)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setFont(theme.mono_font(10))
        bar.setSizeGripEnabled(False)
        self.info_label = QLabel("", bar)
        self.info_label.setFont(theme.mono_font(10))
        self.info_label.setProperty("role", "dim")
        self.info_label.setContentsMargins(0, 0, 14, 0)
        bar.addPermanentWidget(self.info_label)

    def _update_info(self) -> None:
        """Permanent status readout: image count, blend count, canvas size."""
        entries = self.state.entries
        if not entries:
            self.info_label.setText("no images")
            return
        active = self.state.unmuted_count()
        canvas = geometry.target_dimensions([e.full_size for e in entries])
        self.info_label.setText(
            f"{len(entries)} images · {active} in blend · "
            f"canvas {canvas[0]}×{canvas[1]}"
        )

    # -------------------------------------------------------------- notifications

    def _notify(
        self, title: str, text: str, icon: QMessageBox.Icon = QMessageBox.Warning
    ) -> None:
        """Window-modal, non-blocking message box (also records the notice
        so the offscreen selftest can assert on it without an event loop
        stuck in ``exec()``)."""
        self.last_notice = (title, text)
        box = QMessageBox(icon, title, text, QMessageBox.Ok, self)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box.setWindowModality(Qt.WindowModal)
        box.show()

    # -------------------------------------------------------------------- images

    def add_files(self, paths: Sequence[Path]) -> AddReport:
        """Add images (Open toolbar, Finder drop, or programmatic).

        Surfaces the 20-image cap refusal (brief §8) and per-file load
        errors in a clear message.
        """
        report = self.state.add_paths(paths)
        problems: list[str] = []
        if report.refused_cap:
            names = ", ".join(p.name for p in report.refused_cap)
            problems.append(
                f"BlendStack blends at most {engine.MAX_IMAGES} images — "
                f"not added: {names}."
            )
        for path, reason in report.errors:
            problems.append(f"Could not load “{path.name}”: {reason}.")
        if problems:
            self._notify("Some images were not added", "\n\n".join(problems))
        return report

    def _open_dialog(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "Open images", str(self._last_dir), _OPEN_FILTER
        )
        if paths:
            self._last_dir = Path(paths[0]).parent
            self.add_files([Path(p) for p in paths])

    def _sync_strip(self) -> None:
        self.strip.sync(self.state.entries)
        self._update_info()

    # ---------------------------------------------------------- selection wiring

    def _on_selection_changed(self, entry_id: Optional[int]) -> None:
        entry = None if entry_id is None else self.state.entry(entry_id)
        self.adjustments_panel.setEnabled(entry is not None)
        if entry is not None:
            self.adjustments_panel.set_values(entry.adjustments)

    def _on_panel_edited(self, adjustments: Adjustments) -> None:
        entry_id = self.strip.current_entry_id()
        entry = None if entry_id is None else self.state.entry(entry_id)
        if entry is not None:
            # Placement belongs to the canvas Move tool and mute to the strip:
            # whatever the panel emits, never let it clobber them.
            adjustments = replace(
                adjustments,
                move_x=entry.adjustments.move_x,
                move_y=entry.adjustments.move_y,
                mute=entry.adjustments.mute,
            )
            self.state.set_adjustments(entry_id, adjustments)

    def _on_state_adjustments(self, entry_id: int) -> None:
        entry = self.state.entry(entry_id)
        if entry is None:
            return
        self.strip.set_muted(entry_id, entry.adjustments.mute)
        self._update_info()
        # Keep the panel in sync if state changed underneath it (preset load).
        if entry_id == self.strip.current_entry_id():
            current = entry.adjustments
            shown = replace(
                self.adjustments_panel.values(),
                move_x=current.move_x, move_y=current.move_y, mute=current.mute,
            )
            if shown != current:  # skip pure placement / mute changes
                self.adjustments_panel.set_values(current)

    # ------------------------------------------------------------ mute / solo / tools

    def _on_mute_requested(self, entry_id: int, muted: bool) -> None:
        """Strip M button / Shift+M / context menu → document state."""
        self.state.set_mute(entry_id, muted)
        entry = self.state.entry(entry_id)
        if entry is not None:
            active = self.state.unmuted_count()
            verb = "Muted" if muted else "Un-muted"
            self._hint(f"{verb} {entry.path.name} — {active} image"
                       f"{'' if active == 1 else 's'} in the blend.")

    def _toggle_selected_mute(self) -> None:
        entry_id = self.strip.current_entry_id()
        if entry_id is not None:
            self._on_mute_requested(entry_id, not self.state.is_muted(entry_id))

    def _on_solo_changed(self, entry_id: Optional[int]) -> None:
        self.strip.set_solo_id(entry_id)
        entry = None if entry_id is None else self.state.entry(entry_id)
        self.canvas.set_solo_label(None if entry is None else entry.path.name)

    def _on_tool_toggled(self, on: bool) -> None:
        """Move / Crop tool on = full canvas + mouse tools; off = cropped
        result.  Turning it off while cropping cancels the crop edit."""
        if not on:
            self._cancel_crop()
        self.canvas.set_tool_enabled(on)
        self.canvas.set_applied_crop(self.state.crop)
        self.preview.set_show_full_canvas(on)
        if on:
            self._hint(
                "Move / Crop: drag to move the selected image; "
                "double-click the canvas to crop."
            )

    def _hint(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    def _on_move_started(self) -> None:
        entry_id = self.strip.current_entry_id()
        entry = None if entry_id is None else self.state.entry(entry_id)
        if entry is None:
            self._move_base = None
            self._hint("Select an image in the list to move it.")
            return
        self._move_base = (
            entry_id, entry.adjustments.move_x, entry.adjustments.move_y
        )

    def _on_move_dragged(self, dx: float, dy: float) -> None:
        if self._move_base is None:
            return
        entry_id, base_x, base_y = self._move_base
        self.state.set_placement(
            entry_id,
            min(max(base_x + dx, -1.0), 1.0),
            min(max(base_y + dy, -1.0), 1.0),
        )

    def _on_move_finished(self) -> None:
        self._move_base = None  # the layer stays anchored where it was dropped

    def _on_crop_changed(self) -> None:
        self.canvas.set_applied_crop(self.state.crop)
        self.reset_crop_button.setVisible(self.state.crop is not None)

    def _on_crop_mode_changed(self, on: bool) -> None:
        self.crop_bar.setVisible(on)
        self.reset_crop_button.setVisible(self.state.crop is not None)
        for shortcut in self._crop_shortcuts:
            shortcut.setEnabled(on)
        if on:
            self._hint("Crop: drag the handles or the box; Enter = Apply, "
                       "Esc = Cancel.")

    def _apply_crop(self) -> None:
        crop = self.canvas.crop_rect()
        if crop is None:
            return
        self.state.set_crop(None if canvas_tools.is_full_crop(crop) else crop)
        self.canvas.exit_crop_mode()

    def _cancel_crop(self) -> None:
        self.canvas.exit_crop_mode()

    def _reset_crop(self) -> None:
        self.state.set_crop(None)
        self.canvas.exit_crop_mode()

    # ---------------------------------------------------------------- preview I/O

    def _on_preview_ready(
        self, composite: np.ndarray, histogram: np.ndarray, full_canvas: bool
    ) -> None:
        self.canvas.set_composite(composite, full_canvas)
        self.histogram.set_data(histogram)
        h, w = composite.shape[:2]
        kind = ("SOLO" if self.state.solo_id is not None
                else "full canvas" if full_canvas else "cropped")
        self.preview_info.setText(f"{w}×{h} px · {kind}")

    def _on_preview_cleared(self, message: str) -> None:
        self.canvas.clear(message)
        self.histogram.set_data(None)
        self.preview_info.setText("")

    # -------------------------------------------------------------------- presets

    def save_preset_to(self, path: Path) -> Path:
        """Write the current document as a ``.bsp`` preset (brief §5)."""
        return presets.save_preset(
            path,
            mode=self.state.mode,
            params=self.state.params,
            output_format=self.state.output_format,
            images=[(e.path, e.adjustments) for e in self.state.entries],
            crop=self.state.crop,
        )

    def load_preset_from(self, path: Path) -> AddReport:
        """Load a preset; warn per-file about missing images, load the rest."""
        data = presets.load_preset(path)
        missing = [p for p, _ in data["images"] if not p.exists()]
        if missing:
            lines = "\n".join(f"• {p}" for p in missing)
            self._notify(
                "Preset images missing",
                "These files from the preset no longer exist and were "
                f"skipped:\n{lines}",
            )
        keep = [(p, a) for p, a in data["images"] if p.exists()]
        return self.state.restore(
            data["mode"], data["params"], data["output_format"], keep,
            crop=data["crop"],
        )

    def _save_preset_dialog(self) -> None:
        default = self._last_dir / f"blend{presets.PRESET_EXTENSION}"
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save preset", str(default),
            f"BlendStack preset (*{presets.PRESET_EXTENSION})",
        )
        if not path:
            return
        self._last_dir = Path(path).parent
        try:
            self.save_preset_to(Path(path))
        except OSError as exc:
            self._notify("Preset not saved", str(exc))

    def _load_preset_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load preset", str(self._last_dir),
            f"BlendStack preset (*{presets.PRESET_EXTENSION});;All files (*)",
        )
        if not path:
            return
        self._last_dir = Path(path).parent
        try:
            self.load_preset_from(Path(path))
        except presets.PresetError as exc:
            self._notify("Preset not loaded", str(exc))

    # --------------------------------------------------------------------- export

    def _export_ready(self) -> bool:
        """Friendly pre-check: enough images, and enough of them un-muted."""
        if len(self.state.entries) < engine.MIN_IMAGES:
            self._notify(
                "Nothing to export",
                f"Add at least {engine.MIN_IMAGES} images before exporting.",
            )
            return False
        if self.state.unmuted_count() < engine.MIN_IMAGES:
            self._notify(
                "Nothing to export",
                f"Un-mute at least {engine.MIN_IMAGES} images before exporting.",
            )
            return False
        return True

    def _export_dialog(self) -> None:
        if self._export_thread is None and not self._export_ready():
            return
        default_name = bs_io.default_filename(
            self.state.mode, self.state.output_format
        )
        filters = ";;".join(text for text, _fmt in _EXPORT_FILTERS)
        initial_filter = next(
            (t for t, f in _EXPORT_FILTERS if f == self.state.output_format),
            _EXPORT_FILTERS[0][0],
        )
        path, selected = QFileDialog.getSaveFileName(
            self, "Export blend", str(self._last_dir / default_name),
            filters, initial_filter,
        )
        if not path:
            return
        fmt = dict(_EXPORT_FILTERS).get(selected, "tiff")
        out = Path(path)
        suffix_fmt = {
            ".tif": "tiff", ".tiff": "tiff", ".png": "png",
            ".jpg": "jpeg", ".jpeg": "jpeg",
        }.get(out.suffix.lower())
        if suffix_fmt != fmt:  # make the chosen filter authoritative
            out = out.with_suffix("." + bs_io.OUTPUT_FORMATS[fmt])
        self._last_dir = out.parent
        self.export_to(out, fmt)

    def export_to(self, out_path: Path, out_format: str) -> bool:
        """Start a full-resolution export on a background thread.

        Returns True if the export was started.  Completion is announced
        via a dialog and the :attr:`export_done` signal.  Muted images are
        left out (and never loaded); fewer than 2 un-muted images is refused
        with a friendly message.
        """
        if self._export_thread is not None:
            self._notify("Export in progress",
                         "Wait for the current export to finish.")
            return False
        if not self._export_ready():
            return False
        entries = self.state.entries
        self.state.output_format = out_format

        worker = _ExportWorker(
            paths=[e.path for e in entries],
            mode=self.state.mode,
            params=self.state.params,
            adjustments=[e.adjustments for e in entries],
            out_path=Path(out_path),
            out_format=out_format,
            crop=self.state.crop,
        )
        thread = QThread(self)
        thread.setObjectName("blendstack-export")
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        self._export_worker = worker
        self._export_thread = thread

        progress = QProgressDialog(
            "Exporting full-resolution blend…", "", 0, 0, self
        )
        progress.setCancelButton(None)  # blend_files cannot be interrupted
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Export")
        progress.setMinimumDuration(0)
        progress.show()
        self._export_progress = progress

        thread.start()
        return True

    def _finish_export_thread(self) -> None:
        if self._export_progress is not None:
            self._export_progress.close()
            self._export_progress = None
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait(3000)
            self._export_thread = None
        self._export_worker = None

    @Slot(object)
    def _on_export_finished(self, written: Path) -> None:
        self._finish_export_thread()
        self.last_export_path = Path(written)
        self._notify(
            "Export complete", f"Saved {written}", QMessageBox.Information
        )
        self.export_done.emit(True, str(written))

    @Slot(str)
    def _on_export_failed(self, message: str) -> None:
        self._finish_export_thread()
        self._notify("Export failed", message)
        self.export_done.emit(False, message)

    # ------------------------------------------------------------- window events

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = urls_to_supported_paths(event.mimeData().urls())
        if paths:
            self.add_files(paths)
        event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.preview.stop()
        self._finish_export_thread()
        super().closeEvent(event)
