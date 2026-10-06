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

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from blendstack.core import engine

from . import theme
from .knob import Knob, make_section_title, make_segmented, refresh_layouts

__all__ = ["BlendControls", "contrast_pairs"]


def contrast_pairs() -> list[tuple[str, str, str]]:
    """(name, foreground, background) pairs hard-wired in this module."""
    return [
        ("blend hint on panel", theme.TEXT_DIM, theme.PANEL),
        ("blend basis caption on panel", theme.TEXT_DIM, theme.PANEL),
        ("blend combo text on inset", theme.TEXT, theme.INSET),
        ("blend title on panel", theme.GOLD[0], theme.PANEL),
    ]


class BlendControls(QFrame):
    """Mode + softness + bias + comparison basis."""

    mode_changed = Signal(str)          # registry mode name
    param_changed = Signal(str, object)  # parameter name, new value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        self._compact = False

        self.title_label = make_section_title("Blend", self)

        self.mode_combo = QComboBox(self)
        for name in engine.mode_names():
            self.mode_combo.addItem(engine.get_mode(name).label, name)
        self.mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mode_combo.setMinimumContentsLength(12)
        self.mode_combo.setFixedHeight(24)
        self.mode_combo.setToolTip("Blend mode")
        self.mode_combo.setStyleSheet(
            f"QComboBox:disabled {{ color: {theme.TEXT_DIM}; }}"
        )

        self.softness_row = Knob("SOFTNESS", 0, 100, default=0.0)
        self.bias_row = Knob("BIAS", -100, 100, default=0.0)

        # Comparison-basis pill toggle.  The two buttons keep the old
        # radio-button names and are real checkable buttons, so
        # setChecked / isChecked / toggled all work.
        self._basis_box, buttons, self._basis_group = make_segmented(self, ["PER-CH", "LUMA"])
        self.per_channel_radio, self.luminance_radio = buttons
        self.per_channel_radio.setToolTip("Compare each colour channel separately")
        self.luminance_radio.setToolTip("Compare pixels by luminance")
        self.per_channel_radio.setChecked(True)

        # The basis toggle lives in its own container so it can be
        # shown/hidden as a unit alongside the knobs.
        self.basis_row = QWidget(self)
        basis_layout = QHBoxLayout(self.basis_row)
        basis_layout.setContentsMargins(0, 0, 0, 0)
        basis_layout.setSpacing(6)
        self._basis_caption = QLabel("BASIS", self.basis_row)
        self._basis_caption.setProperty("role", "dim")
        cap_font = theme.label_font(8, bold=True)
        cap_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
        self._basis_caption.setFont(cap_font)
        basis_layout.addWidget(self._basis_caption)
        basis_layout.addWidget(self._basis_box)
        basis_layout.addStretch(1)

        # Short hint shown for modes that declare no parameters.
        self.hint_label = QLabel("Strength = per-image opacity", self)
        self.hint_label.setProperty("role", "dim")
        self.hint_label.setWordWrap(True)
        self.hint_label.setFont(theme.label_font(10))

        self.knob_row = QWidget(self)
        knob_layout = QHBoxLayout(self.knob_row)
        knob_layout.setContentsMargins(0, 0, 0, 0)
        knob_layout.setSpacing(4)
        knob_layout.addStretch(1)
        knob_layout.addWidget(self.softness_row)
        knob_layout.addWidget(self.bias_row)
        knob_layout.addStretch(1)

        # Map each declared parameter name to the widget that controls it.
        # Visibility is driven by iterating the current mode's declared
        # parameter names, so a future param widget just needs an entry here.
        self._param_widgets: dict[str, QWidget] = {
            "softness": self.softness_row,
            "bias": self.bias_row,
            "basis": self.basis_row,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.knob_row)
        layout.addWidget(self.basis_row)
        layout.addWidget(self.hint_label)
        layout.addStretch(1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.softness_row.valueChanged.connect(
            lambda v: self.param_changed.emit("softness", float(v))
        )
        self.bias_row.valueChanged.connect(
            lambda v: self.param_changed.emit("bias", float(v))
        )
        self.per_channel_radio.toggled.connect(self._on_basis)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._sync_visibility(self.mode_combo.currentData())

    # ----------------------------------------------------------------- sizing

    def set_compact(self, compact: bool) -> None:
        """Use the smaller knob dial (rack on a short / narrow window)."""
        self._compact = bool(compact)
        size = Knob.COMPACT if compact else Knob.NORMAL
        for knob in (self.softness_row, self.bias_row):
            knob.set_dial_size(size)
        refresh_layouts(self)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(190, super().minimumSizeHint().height())

    # ----------------------------------------------------------------- logic

    def _on_mode(self, _index: int) -> None:
        self._sync_visibility(self.mode_combo.currentData())
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
        self.knob_row.setVisible(bool(declared & {"softness", "bias"}))
        self.hint_label.setVisible(not (declared & set(self._param_widgets)))

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
