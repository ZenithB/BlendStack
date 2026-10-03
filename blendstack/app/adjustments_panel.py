"""Per-image adjustments panel (project brief §5 layout, §4.1 definitions).

Edits the :class:`~blendstack.core.adjustments.Adjustments` of the image
currently selected in the strip: exposure -3...+3 EV, contrast / saturation
-100...+100, tone curves (master + R/G/B), sharpen radius 0.5-20 px, sharpen
amount 0-200 %, noise removal 0-100 %, opacity 0-100 %, the per-image
processing order, plus a Reset button.  Emits :attr:`adjustments_edited`
with a fresh frozen ``Adjustments`` on every user change; the main window
routes it to the document state for the selected entry.

The panel does not own every field of ``Adjustments`` (layer placement
``move_x`` / ``move_y`` belongs to the canvas tool).  It therefore remembers
the last ``Adjustments`` it was given / emitted and builds every result with
:func:`dataclasses.replace` on it, so fields it does not edit are preserved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from blendstack.core.adjustments import Adjustments, IDENTITY_CURVE

from .curve_editor import CurveEditor
from .order_panel import OrderPanel
from .slider_row import SliderRow

__all__ = ["AdjustmentsPanel"]


class _Section(QWidget):
    """A header button (with expand/collapse arrow) above a content widget."""

    def __init__(self, title: str, content: QWidget, expanded: bool = True,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.content = content
        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        self.toggle.toggled.connect(self._on_toggled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle, 0, Qt.AlignLeft)
        layout.addWidget(content)
        self.toggle.setChecked(expanded)
        self._on_toggled(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)


def _curve_is_identity(points) -> bool:
    return tuple(tuple(p) for p in points) == tuple(tuple(p) for p in IDENTITY_CURVE)


class AdjustmentsPanel(QGroupBox):
    """Sliders, curves and processing order for the selected image."""

    #: Emitted with the new Adjustments after any user edit.
    adjustments_edited = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Image adjustments", parent)
        # Slider integer ranges map onto brief §4.1 UI ranges via `scale`.
        self.exposure_row = SliderRow("Exposure", -300, 300, scale=0.01,
                                      decimals=2, suffix=" EV")
        self.contrast_row = SliderRow("Contrast", -100, 100)
        self.saturation_row = SliderRow("Saturation", -100, 100)
        self.radius_row = SliderRow("Sharpen radius", 5, 200, scale=0.1,
                                    decimals=1, suffix=" px")
        self.amount_row = SliderRow("Sharpen amount", 0, 200, suffix=" %")
        self.denoise_row = SliderRow("Noise removal", 0, 100, suffix=" %")
        self.opacity_row = SliderRow("Opacity", 0, 100, suffix=" %")
        self.curve_editor = CurveEditor(self)
        self.order_panel = OrderPanel(self)
        self.curves_section = _Section("Curves", self.curve_editor, True, self)
        self.order_section = _Section("Routing", self.order_panel, True, self)
        self.reset_button = QPushButton("Reset", self)

        self._base = Adjustments()
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.addWidget(self.exposure_row)
        layout.addWidget(self.contrast_row)
        layout.addWidget(self.saturation_row)
        layout.addWidget(self.curves_section)
        layout.addWidget(self.radius_row)
        layout.addWidget(self.amount_row)
        layout.addWidget(self.denoise_row)
        layout.addWidget(self.opacity_row)
        layout.addWidget(self.order_section)
        layout.addWidget(self.reset_button)

        for row in self._rows():
            row.valueChanged.connect(self._emit_edited)
        self.curve_editor.curves_changed.connect(self._emit_edited)
        self.order_panel.order_changed.connect(self._emit_edited)
        self.reset_button.clicked.connect(self.reset)

        self.set_values(Adjustments())
        self.setEnabled(False)  # until an image is selected

    def _rows(self) -> tuple[SliderRow, ...]:
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
    def _read(row: SliderRow, base_value: float) -> float:
        """Slider value, but keep ``base_value`` untouched while the slider
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
        layer placement; then emit."""
        keep = self._base
        self.set_values(replace(Adjustments(), move_x=keep.move_x, move_y=keep.move_y))
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
