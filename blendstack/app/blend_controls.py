"""Global blend controls panel (project brief §5 layout, §4.2 parameters).

Mode dropdown (display text = ``engine.get_mode(name).label``, data = the
registry name), softness 0–100, bias −100…+100, and the comparison-basis
toggle (per-channel / luminance).  Emits :attr:`mode_changed` and
:attr:`param_changed`; the main window routes them into the document state,
where a change re-runs only the fold (the per-image adjusted-proxy caches
stay valid — brief §5 live-preview strategy).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from blendstack.core import engine

from .slider_row import SliderRow

__all__ = ["BlendControls"]


class BlendControls(QGroupBox):
    """Mode + softness + bias + comparison basis."""

    mode_changed = Signal(str)          # registry mode name
    param_changed = Signal(str, object)  # parameter name, new value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Blend", parent)

        self.mode_combo = QComboBox(self)
        for name in engine.mode_names():
            self.mode_combo.addItem(engine.get_mode(name).label, name)

        self.softness_row = SliderRow("Softness", 0, 100)
        self.bias_row = SliderRow("Bias", -100, 100)

        self.per_channel_radio = QRadioButton("Per-channel", self)
        self.luminance_radio = QRadioButton("Luminance", self)
        self.per_channel_radio.setChecked(True)
        self._basis_group = QButtonGroup(self)
        self._basis_group.addButton(self.per_channel_radio)
        self._basis_group.addButton(self.luminance_radio)

        basis_row = QHBoxLayout()
        basis_row.addWidget(QLabel("Basis:", self))
        basis_row.addWidget(self.per_channel_radio)
        basis_row.addWidget(self.luminance_radio)
        basis_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.softness_row)
        layout.addWidget(self.bias_row)
        layout.addLayout(basis_row)

        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.softness_row.valueChanged.connect(
            lambda v: self.param_changed.emit("softness", float(v))
        )
        self.bias_row.valueChanged.connect(
            lambda v: self.param_changed.emit("bias", float(v))
        )
        self.per_channel_radio.toggled.connect(self._on_basis)

    def _on_mode(self, _index: int) -> None:
        self.mode_changed.emit(self.mode_combo.currentData())

    def _on_basis(self, per_channel_checked: bool) -> None:
        self.param_changed.emit(
            "basis", "per_channel" if per_channel_checked else "luminance"
        )

    def set_from_state(self, mode: str, params: Mapping[str, Any]) -> None:
        """Sync widgets from the document state without emitting changes."""
        index = self.mode_combo.findData(mode)
        if index >= 0:
            blocked = self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(blocked)
        self.softness_row.set_value(float(params.get("softness", 0.0)))
        self.bias_row.set_value(float(params.get("bias", 0.0)))
        basis = str(params.get("basis", "per_channel"))
        target = (
            self.per_channel_radio if basis == "per_channel"
            else self.luminance_radio
        )
        blocked = self.per_channel_radio.blockSignals(True)
        blocked2 = self.luminance_radio.blockSignals(True)
        target.setChecked(True)
        self.per_channel_radio.blockSignals(blocked)
        self.luminance_radio.blockSignals(blocked2)
