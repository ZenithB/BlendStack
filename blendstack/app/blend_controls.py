"""Global blend controls panel (project brief §5 layout, §4.2 parameters).

Mode dropdown (display text = ``engine.get_mode(name).label``, data = the
registry name), plus the parameter widgets each mode declares: softness
0–100, bias −100…+100, and the comparison-basis toggle (per-channel /
luminance).  The parameter widgets are shown/hidden **per selected mode** —
only the widgets whose parameter names the current mode declares (via
``engine.get_mode(name).parameters``) are visible.  The two Canon modes
declare softness + bias + basis; the newer modes declare none, so for them
only the dropdown shows and the per-image opacity is the strength control.

Emits :attr:`mode_changed` and :attr:`param_changed`; the main window
routes them into the document state, where a change re-runs only the fold
(the per-image adjusted-proxy caches stay valid — brief §5 live-preview
strategy).
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

        # The basis toggle lives in its own container widget so it can be
        # shown/hidden as a unit alongside the slider rows.
        self.basis_row = QWidget(self)
        basis_layout = QHBoxLayout(self.basis_row)
        basis_layout.setContentsMargins(0, 0, 0, 0)
        basis_layout.addWidget(QLabel("Basis:", self.basis_row))
        basis_layout.addWidget(self.per_channel_radio)
        basis_layout.addWidget(self.luminance_radio)
        basis_layout.addStretch(1)

        # Map each declared parameter name to the widget that controls it.
        # Visibility is driven by iterating the current mode's declared
        # parameter names, so a future param widget just needs an entry here.
        self._param_widgets: dict[str, QWidget] = {
            "softness": self.softness_row,
            "bias": self.bias_row,
            "basis": self.basis_row,
        }

        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.softness_row)
        layout.addWidget(self.bias_row)
        layout.addWidget(self.basis_row)

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

    def _sync_visibility(self, mode: str) -> None:
        """Show only the parameter widgets the given mode declares."""
        declared = {p.name for p in engine.get_mode(mode).parameters}
        for name, widget in self._param_widgets.items():
            widget.setVisible(name in declared)

    def set_from_state(self, mode: str, params: Mapping[str, Any]) -> None:
        """Sync widgets from the document state without emitting changes."""
        index = self.mode_combo.findData(mode)
        if index >= 0:
            blocked = self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(blocked)
        self._sync_visibility(mode)
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
