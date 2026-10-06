"""Per-image adjustments panel (project brief §5 layout, §4.1 definitions).

Edits the :class:`~blendstack.core.adjustments.Adjustments` of the image
currently selected in the strip, as one wide horizontal "rack" block with
three titled sections (IMAGE knobs | CURVES | ORDER): exposure -3...+3 EV, contrast / saturation
-100...+100, tone curves (master + R/G/B), sharpen radius 0.5-20 px, sharpen
amount 0-200 %, noise removal 0-100 %, opacity 0-100 %, the per-image
processing order, plus a Reset button.  Emits :attr:`adjustments_edited`
with a fresh frozen ``Adjustments`` on every user change; the main window
routes it to the document state for the selected entry.

The panel does not own every field of ``Adjustments`` (layer placement
``move_x`` / ``move_y`` belongs to the canvas tool, ``mute`` to the strip).  It therefore remembers
the last ``Adjustments`` it was given / emitted and builds every result with
:func:`dataclasses.replace` on it, so fields it does not edit are preserved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import Adjustments, IDENTITY_CURVE

from .curve_editor import CurveEditor
from .knob import (
    Knob,
    ghost_button_stylesheet,
    make_section_title,
    make_vrule,
    refresh_layouts,
)
from .order_panel import OrderPanel

__all__ = ["AdjustmentsPanel"]


def _section(title: str, content: QWidget, parent: QWidget) -> QWidget:
    """A titled column of the rack: gold caption above ``content``."""
    box = QWidget(parent)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addWidget(make_section_title(title, box), 0, Qt.AlignLeft)
    lay.addWidget(content)
    lay.addStretch(1)
    return box


def _curve_is_identity(points) -> bool:
    return tuple(tuple(p) for p in points) == tuple(tuple(p) for p in IDENTITY_CURVE)


class AdjustmentsPanel(QFrame):
    """Knobs, curves and processing order for the selected image, laid out as
    one wide rack block: IMAGE | CURVES | ORDER."""

    #: Emitted with the new Adjustments after any user edit.
    adjustments_edited = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        # Knob integer ranges map onto brief §4.1 UI ranges via `scale`.
        self.exposure_row = Knob("EXPOSURE", -300, 300, scale=0.01,
                                 decimals=2, suffix=" EV", default=0.0)
        self.contrast_row = Knob("CONTRAST", -100, 100, default=0.0)
        self.saturation_row = Knob("SATURATION", -100, 100, default=0.0)
        self.opacity_row = Knob("OPACITY", 0, 100, suffix=" %", default=100.0)
        self.radius_row = Knob("SHARPEN RADIUS", 5, 200, scale=0.1,
                               decimals=1, suffix=" px", default=1.0)
        self.amount_row = Knob("SHARPEN AMOUNT", 0, 200, suffix=" %", default=0.0)
        self.denoise_row = Knob("NOISE REMOVAL", 0, 100, suffix=" %", default=0.0)
        self.curve_editor = CurveEditor(self)
        self.order_panel = OrderPanel(self)
        self.reset_button = QPushButton("RESET", self)
        self.reset_button.setProperty("variant", "ghost")
        self.reset_button.setStyleSheet(ghost_button_stylesheet())
        self.reset_button.setFixedHeight(22)
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.setToolTip(
            "Reset exposure, contrast, saturation, opacity, sharpen, noise removal, "
            "curves and order (placement and mute are kept)"
        )

        self._base = Adjustments()
        self._syncing = False
        self._compact = False
        self._auto_compact = True

        # IMAGE section: 4 x 2 grid of knobs, RESET in the last cell.
        image = QWidget(self)
        grid = QGridLayout(image)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(4)
        cells = (
            self.exposure_row, self.contrast_row, self.saturation_row, self.opacity_row,
            self.radius_row, self.amount_row, self.denoise_row,
        )
        for i, knob in enumerate(cells):
            grid.addWidget(knob, i // 4, i % 4, Qt.AlignHCenter | Qt.AlignTop)
        grid.addWidget(self.reset_button, 1, 3, Qt.AlignCenter)
        self.image_grid = image
        self._grid = grid

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)
        layout.addWidget(_section("Image", image, self), 0)
        layout.addWidget(make_vrule(self))
        layout.addWidget(_section("Curves", self.curve_editor, self), 0)
        layout.addWidget(make_vrule(self))
        layout.addWidget(_section("Order", self.order_panel, self), 0)

        for row in self._rows():
            row.valueChanged.connect(self._emit_edited)
        self.curve_editor.curves_changed.connect(self._emit_edited)
        self.order_panel.order_changed.connect(self._emit_edited)
        self.reset_button.clicked.connect(self.reset)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Measure the normal and the compact layout once so sizeHint /
        # minimumSizeHint are stable and compact mode can switch on width.
        self._compact_hint = QSize()
        self._normal_hint = QSize()
        self._measure_hints()

        self.set_values(Adjustments())
        self.setEnabled(False)  # until an image is selected

    # ----------------------------------------------------------------- sizing

    def _measure_hints(self) -> None:
        self._apply_compact(True)
        self._compact_hint = self.layout().sizeHint()
        self._apply_compact(False)
        self._normal_hint = self.layout().sizeHint()

    def _apply_compact(self, compact: bool) -> None:
        self._compact = bool(compact)
        size = Knob.COMPACT if compact else Knob.NORMAL
        for knob in self._rows():
            knob.set_dial_size(size)
        self.curve_editor.set_compact(compact)
        refresh_layouts(self)

    def set_compact(self, compact: bool, auto: bool = False) -> None:
        """Switch to the compact rack (smaller dials / plot).  With
        ``auto=False`` (default) this also turns off width-driven switching."""
        if not auto:
            self._auto_compact = False
        if bool(compact) != self._compact:
            self._apply_compact(compact)

    def is_compact(self) -> bool:
        return self._compact

    def sizeHint(self) -> QSize:  # noqa: N802
        return self._normal_hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self._compact_hint

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_compact:
            want = self.width() < self._normal_hint.width()
            if want != self._compact:
                self._apply_compact(want)

    def _rows(self) -> tuple[Knob, ...]:
        return (
            self.exposure_row,
            self.contrast_row,
            self.saturation_row,
            self.radius_row,
            self.amount_row,
            self.denoise_row,
            self.opacity_row,
        )

    @staticmethod
    def _read(row: Knob, base_value: float) -> float:
        """Knob value, but keep ``base_value`` untouched while the knob
        still sits on its step (so un-edited fields never get re-quantised)."""
        return base_value if row.matches(base_value) else row.value()

    def values(self) -> Adjustments:
        """Read the widgets into a frozen ``Adjustments``.  Fields the panel
        does not edit (move_x / move_y) come from the last set/emitted value."""
        b = self._base
        master, red, green, blue = self.curve_editor.curves()
        return replace(
            b,
            exposure=self._read(self.exposure_row, b.exposure),
            contrast=self._read(self.contrast_row, b.contrast),
            saturation=self._read(self.saturation_row, b.saturation),
            sharpen_radius=self._read(self.radius_row, b.sharpen_radius),
            sharpen_amount=self._read(self.amount_row, b.sharpen_amount),
            denoise=self._read(self.denoise_row, b.denoise),
            opacity=self._read(self.opacity_row, b.opacity),
            curve_master=master,
            curve_red=red,
            curve_green=green,
            curve_blue=blue,
            order=self.order_panel.order(),
        )

    def set_values(self, adjustments: Adjustments) -> None:
        """Sync every widget from state without emitting edits."""
        self._syncing = True
        try:
            self._base = adjustments
            self.exposure_row.set_value(adjustments.exposure)
            self.contrast_row.set_value(adjustments.contrast)
            self.saturation_row.set_value(adjustments.saturation)
            self.radius_row.set_value(adjustments.sharpen_radius)
            self.amount_row.set_value(adjustments.sharpen_amount)
            self.denoise_row.set_value(adjustments.denoise)
            self.opacity_row.set_value(adjustments.opacity)
            self.curve_editor.set_curves(
                adjustments.curve_master, adjustments.curve_red,
                adjustments.curve_green, adjustments.curve_blue,
            )
            self.order_panel.set_order(adjustments.order)
            self._update_active(adjustments)
        finally:
            self._syncing = False

    def reset(self) -> None:
        """Reset button (brief §5): every field the panel edits back to its
        default (identity curves, default order, opacity 100), keeping the
        layer placement and the mute flag; then emit."""
        keep = self._base
        self.set_values(replace(Adjustments(), move_x=keep.move_x, move_y=keep.move_y,
                                mute=keep.mute))
        self._emit_edited()

    def _update_active(self, a: Adjustments) -> None:
        active = set()
        if a.exposure != 0.0:
            active.add("exposure")
        if a.contrast != 0.0:
            active.add("contrast")
        if not all(_curve_is_identity(c) for c in (
                a.curve_master, a.curve_red, a.curve_green, a.curve_blue)):
            active.add("curves")
        if a.saturation != 0.0:
            active.add("saturation")
        if a.denoise != 0.0:
            active.add("denoise")
        if a.sharpen_amount != 0.0:
            active.add("sharpen")
        self.order_panel.set_active_stages(active)

    def _emit_edited(self, *_ignored: object) -> None:
        if self._syncing:
            return
        adjustments = self.values()
        self._base = adjustments
        self._update_active(adjustments)
        self.adjustments_edited.emit(adjustments)
