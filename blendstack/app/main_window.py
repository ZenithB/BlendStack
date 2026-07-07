"""Main window — assembles the standalone app (project brief §5).

Layout: left = reorderable image strip (fold order, top = base image);
centre = live preview canvas; right = per-image adjustments panel, global
blend controls and the composite histogram.  Toolbar: Open, Save Preset,
Load Preset, Export.

Export (brief §5 / §4.4) runs the **full-resolution** pipeline via
``engine.blend_files`` (streams one file at a time, memory-bounded) on a
background thread with a modal indeterminate progress dialog — the UI
thread is never blocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from blendstack.core import engine
from blendstack.core import io as bs_io
from blendstack.core.adjustments import Adjustments

from . import presets
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
    ) -> None:
        super().__init__()
        self._paths = list(paths)
        self._mode = mode
        self._params = dict(params)
        self._adjustments = list(adjustments)
        self._out_path = out_path
        self._out_format = out_format

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
            )
        except Exception as exc:  # noqa: BLE001 — surfaced in a dialog
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(written)


class MainWindow(QMainWindow):
    """The BlendStack standalone app window (brief §5)."""

    #: (success, message) — emitted when a background export completes.
    export_done = Signal(bool, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BlendStack")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.state = DocumentState(self)
        #: Last user-facing notice as (title, text) — inspected by selftest.
        self.last_notice: Optional[tuple[str, str]] = None
        self.last_export_path: Optional[Path] = None
        self._last_dir = Path.home()
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[_ExportWorker] = None
        self._export_progress: Optional[QProgressDialog] = None

        # -- widgets ---------------------------------------------------------
        self.strip = ImageStrip(self)
        self.canvas = PreviewCanvas(self)
        self.adjustments_panel = AdjustmentsPanel(self)
        self.blend_controls = BlendControls(self)
        self.histogram = HistogramWidget(self)

        strip_column = QVBoxLayout()
        strip_label = QLabel("Images — fold order (top = base)", self)
        strip_label.setWordWrap(True)
        strip_column.addWidget(strip_label)
        strip_column.addWidget(self.strip, 1)

        right_column = QVBoxLayout()
        right_column.addWidget(self.adjustments_panel)
        right_column.addWidget(self.blend_controls)
        right_column.addStretch(1)
        right_column.addWidget(QLabel("Composite histogram", self))
        right_column.addWidget(self.histogram)
        right = QWidget(self)
        right.setLayout(right_column)
        right.setFixedWidth(300)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addLayout(strip_column)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(right)
        self.setCentralWidget(central)

        self._build_toolbar()

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

        # -- wiring: UI -> state ------------------------------------------------
        self.strip.files_dropped.connect(self.add_files)
        self.strip.order_changed.connect(self.state.reorder)
        self.strip.remove_requested.connect(self.state.remove)
        self.strip.selection_changed.connect(self._on_selection_changed)
        self.adjustments_panel.adjustments_edited.connect(self._on_panel_edited)
        self.blend_controls.mode_changed.connect(self.state.set_mode)
        self.blend_controls.param_changed.connect(self.state.set_param)

        self.blend_controls.set_from_state(self.state.mode, self.state.params)
        self.canvas.clear()

    # ------------------------------------------------------------------ toolbar

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)

        self.open_action = QAction("Open…", self)
        self.open_action.triggered.connect(self._open_dialog)
        self.save_preset_action = QAction("Save Preset…", self)
        self.save_preset_action.triggered.connect(self._save_preset_dialog)
        self.load_preset_action = QAction("Load Preset…", self)
        self.load_preset_action.triggered.connect(self._load_preset_dialog)
        self.export_action = QAction("Export…", self)
        self.export_action.triggered.connect(self._export_dialog)
        for action in (
            self.open_action,
            self.save_preset_action,
            self.load_preset_action,
            self.export_action,
        ):
            toolbar.addAction(action)

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

    # ---------------------------------------------------------- selection wiring

    def _on_selection_changed(self, entry_id: Optional[int]) -> None:
        entry = None if entry_id is None else self.state.entry(entry_id)
        self.adjustments_panel.setEnabled(entry is not None)
        if entry is not None:
            self.adjustments_panel.set_values(entry.adjustments)

    def _on_panel_edited(self, adjustments: Adjustments) -> None:
        entry_id = self.strip.current_entry_id()
        if entry_id is not None:
            self.state.set_adjustments(entry_id, adjustments)

    def _on_state_adjustments(self, entry_id: int) -> None:
        # Keep the panel in sync if state changed underneath it (preset load).
        if entry_id == self.strip.current_entry_id():
            entry = self.state.entry(entry_id)
            if entry is not None:
                self.adjustments_panel.set_values(entry.adjustments)

    # ---------------------------------------------------------------- preview I/O

    def _on_preview_ready(self, composite: np.ndarray, histogram: np.ndarray) -> None:
        self.canvas.set_composite(composite)
        self.histogram.set_data(histogram)

    def _on_preview_cleared(self, message: str) -> None:
        self.canvas.clear(message)
        self.histogram.set_data(None)

    # -------------------------------------------------------------------- presets

    def save_preset_to(self, path: Path) -> Path:
        """Write the current document as a ``.bsp`` preset (brief §5)."""
        return presets.save_preset(
            path,
            mode=self.state.mode,
            params=self.state.params,
            output_format=self.state.output_format,
            images=[(e.path, e.adjustments) for e in self.state.entries],
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
            data["mode"], data["params"], data["output_format"], keep
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

    def _export_dialog(self) -> None:
        if len(self.state.entries) < engine.MIN_IMAGES:
            self._notify(
                "Nothing to export",
                f"Add at least {engine.MIN_IMAGES} images before exporting.",
            )
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
        via a dialog and the :attr:`export_done` signal.
        """
        if self._export_thread is not None:
            self._notify("Export in progress",
                         "Wait for the current export to finish.")
            return False
        entries = self.state.entries
        if len(entries) < engine.MIN_IMAGES:
            self._notify(
                "Nothing to export",
                f"Add at least {engine.MIN_IMAGES} images before exporting.",
            )
            return False
        self.state.output_format = out_format

        worker = _ExportWorker(
            paths=[e.path for e in entries],
            mode=self.state.mode,
            params=self.state.params,
            adjustments=[e.adjustments for e in entries],
            out_path=Path(out_path),
            out_format=out_format,
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
