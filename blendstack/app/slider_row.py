"""Shared labelled-slider row used by both right-hand panels (brief §5).

A ``QSlider`` works in integers; :class:`SliderRow` maps the integer range
onto a real value via ``scale`` (real = slider × scale) and shows the live
value next to the label.  ``set_value`` updates the UI silently (used when
syncing from state), while user interaction emits :attr:`valueChanged`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

__all__ = ["SliderRow"]

# The native macOS slider knob (~20 px) overflows the tight row spacing and
# clips into the rows above/below. Style a smaller circular handle (~15 px,
# i.e. ~5 px smaller) over a thin groove. Styling any sub-control switches
# the whole slider to styled rendering, so groove + handle are both defined;
# palette() keeps it correct in light and dark themes.
_HANDLE_PX = 15
_SLIDER_QSS = f"""
QSlider::groove:horizontal {{
    height: 4px;
    background: palette(mid);
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: palette(button);
    border: 1px solid palette(mid);
    width: {_HANDLE_PX}px;
    height: {_HANDLE_PX}px;
    border-radius: {_HANDLE_PX // 2}px;
    margin: -{(_HANDLE_PX - 4) // 2}px 0;
}}
QSlider::handle:horizontal:hover {{
    border-color: palette(highlight);
}}
"""


class SliderRow(QWidget):
    """Label + value readout above a horizontal slider."""

    valueChanged = Signal(float)  # noqa: N815 (Qt signal naming)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        scale: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scale = scale
        self._decimals = decimals
        self._suffix = suffix

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(minimum, maximum)
        self.slider.setStyleSheet(_SLIDER_QSS)
        self._name_label = QLabel(label, self)
        self._value_label = QLabel(self)
        self._value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._name_label)
        header.addStretch(1)
        header.addWidget(self._value_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slider)
        self._refresh_label()

    def value(self) -> float:
        return self.slider.value() * self._scale

    def set_value(self, value: float) -> None:
        """Move the slider without emitting :attr:`valueChanged`."""
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(round(value / self._scale))
        self.slider.blockSignals(blocked)
        self._refresh_label()

    def _on_slider(self, _raw: int) -> None:
        self._refresh_label()
        self.valueChanged.emit(self.value())

    def _refresh_label(self) -> None:
        self._value_label.setText(
            f"{self.value():+.{self._decimals}f}{self._suffix}"
            if self.slider.minimum() < 0
            else f"{self.value():.{self._decimals}f}{self._suffix}"
        )
