"""Per-image adjustments panel (project brief §5 layout, §4.1 definitions).

Edits the :class:`~blendstack.core.adjustments.Adjustments` of the image
currently selected in the strip: exposure −3…+3 EV, contrast −100…+100,
saturation −100…+100, sharpen radius 0.5–10 px, sharpen amount 0–200 %,
opacity 0–100 %, plus a Reset button.  Emits :attr:`adjustments_edited`
with a fresh frozen ``Adjustments`` on every user change; the main window
routes it to the document state for the selected entry.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

from blendstack.core.adjustments import Adjustments

from .slider_row import SliderRow

__all__ = ["AdjustmentsPanel"]


class AdjustmentsPanel(QGroupBox):
    """Sliders for the selected image's pre-blend adjustments."""

    #: Emitted with the new Adjustments after any user edit.
    adjustments_edited = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Image adjustments", parent)
        # Slider integer ranges map onto brief §4.1 UI ranges via `scale`.
        self.exposure_row = SliderRow("Exposure", -300, 300, scale=0.01,
                                      decimals=2, suffix=" EV")
        self.contrast_row = SliderRow("Contrast", -100, 100)
        self.saturation_row = SliderRow("Saturation", -100, 100)
        self.radius_row = SliderRow("Sharpen radius", 5, 100, scale=0.1,
                                    decimals=1, suffix=" px")
        self.amount_row = SliderRow("Sharpen amount", 0, 200, suffix=" %")
        self.opacity_row = SliderRow("Opacity", 0, 100, suffix=" %")
        self.reset_button = QPushButton("Reset", self)

        layout = QVBoxLayout(self)
        for row in self._rows():
            layout.addWidget(row)
        layout.addWidget(self.reset_button)

        for row in self._rows():
            row.valueChanged.connect(self._emit_edited)
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
            self.opacity_row,
        )

    def values(self) -> Adjustments:
        """Read the sliders into a frozen ``Adjustments``."""
        return Adjustments(
            exposure=self.exposure_row.value(),
            contrast=self.contrast_row.value(),
            saturation=self.saturation_row.value(),
            sharpen_radius=self.radius_row.value(),
            sharpen_amount=self.amount_row.value(),
            opacity=self.opacity_row.value(),
        )

    def set_values(self, adjustments: Adjustments) -> None:
        """Sync the sliders from state without emitting edits."""
        self.exposure_row.set_value(adjustments.exposure)
        self.contrast_row.set_value(adjustments.contrast)
        self.saturation_row.set_value(adjustments.saturation)
        self.radius_row.set_value(adjustments.sharpen_radius)
        self.amount_row.set_value(adjustments.sharpen_amount)
        self.opacity_row.set_value(adjustments.opacity)

    def reset(self) -> None:
        """Reset button (brief §5): back to identity settings, and emit."""
        self.set_values(Adjustments())
        self._emit_edited()

    def _emit_edited(self, *_ignored: object) -> None:
        self.adjustments_edited.emit(self.values())
