"""BlendStack visual theme: a dark-grey, techy "synth rack" look.

One module owns every colour, font and the app-wide stylesheet so the whole
UI stays consistent.  Custom-painted widgets (knobs, curve editor, histogram,
strip rows, canvas overlays) import the colour constants from here; ordinary
Qt widgets are styled by :func:`stylesheet`, installed with :func:`apply`.

Palette (supplied by the user) — three ramps, light → dark:

* GOLD   (primary)     FFC437  F3AC00  C48B00  A07100  745200
* PURPLE (secondary 1) 792DAC  6509A3  510784  41046B  2F024E
* GREEN  (secondary 2) 25AD7A  00A466  008553  006C43  004E31

Colour roles
------------
* **Gold** — the primary accent: knob value arcs, selection, section titles,
  focus rings, the primary button, the solo light.
* **Green** — "on / healthy / active": LEDs, enabled-tool indicator, the
  luma/OK states, positive feedback.
* **Purple** — structure: raised panel headers, chips, the mute light and
  other *fills*.  The purples are too dark to read as text on dark grey (~2:1),
  so purple is only ever a fill or border, with white text drawn on top.
* **Neutrals** — dark-grey background, slightly lighter panels, near-black
  insets for displays.

Readability rule (checked by ``selftest_theme``): body text ≥ 7:1 against the
surface it sits on, every other piece of text ≥ 4.5:1 (WCAG AA).

Usage::

    from blendstack.app import theme
    theme.apply(QApplication.instance())      # once, at startup

Stylesheet hooks (set with ``widget.setProperty(...)``):

* ``QPushButton[variant="primary"]`` — filled gold button (e.g. Export)
* ``QPushButton[variant="ghost"]``   — flat button, gold hover
* ``QLabel[role="title"]``  — small, letter-spaced, uppercase section title (gold)
* ``QLabel[role="value"]``  — monospace numeric readout
* ``QLabel[role="dim"]``    — secondary text (still high contrast)
* ``QFrame[panel="true"]``  — a raised panel (background + border + radius)
* ``QFrame[inset="true"]``  — a recessed display area
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

__all__ = [
    "GOLD", "PURPLE", "GREEN",
    "BG", "PANEL", "PANEL_HI", "INSET", "BORDER", "BORDER_HI",
    "TEXT", "TEXT_DIM", "TEXT_ON_GOLD", "TEXT_ON_PURPLE",
    "ACCENT", "ACCENT_HI", "ACCENT_LO", "LED_ON", "MUTE", "SOLO",
    "CHANNEL_R", "CHANNEL_G", "CHANNEL_B", "CHANNEL_L",
    "color", "contrast_ratio", "relative_luminance", "text_pairs",
    "mono_font", "label_font", "title_font", "mono_family", "sans_family", "stylesheet", "apply",
]

# --------------------------------------------------------------------- palette

GOLD = ["#FFC437", "#F3AC00", "#C48B00", "#A07100", "#745200"]
PURPLE = ["#792DAC", "#6509A3", "#510784", "#41046B", "#2F024E"]
GREEN = ["#25AD7A", "#00A466", "#008553", "#006C43", "#004E31"]

# Neutrals — the main background is a dark grey.
BG = "#212225"          # window background
PANEL = "#2a2b30"       # raised panels / rack sections
PANEL_HI = "#34363c"    # hovered / raised-further surfaces
INSET = "#151618"       # recessed displays (canvas, histogram, curve plot)
BORDER = "#43464d"      # panel outlines
BORDER_HI = "#5a5e67"   # stronger outlines

# Text — always high contrast.
TEXT = "#F4F4F6"        # primary text            (≈ 14:1 on BG)
TEXT_DIM = "#C6C8CF"    # secondary text          (≈ 9:1 on BG, ≥ 7:1 on PANEL)
TEXT_ON_GOLD = "#17181a"      # dark text on a gold fill
TEXT_ON_PURPLE = "#FFFFFF"    # white text on a purple fill

# Semantic accents.
ACCENT = GOLD[0]
ACCENT_HI = "#FFD46B"   # lighter gold for hover (derived)
ACCENT_LO = GOLD[3]     # dim gold for tracks / disabled accents
LED_ON = GREEN[0]
SOLO = GOLD[0]
MUTE = PURPLE[0]

# Channel colours keep their meaning (red/green/blue/luma); green comes from
# the palette, red and blue are tuned to sit well with it on dark grey.
CHANNEL_R = "#FF6B6B"
CHANNEL_G = GREEN[0]
CHANNEL_B = "#6C9DFF"
CHANNEL_L = "#F4F4F6"


# --------------------------------------------------------------------- helpers

def color(value: str, alpha: int | None = None) -> QColor:
    """``QColor`` from a ``#RRGGBB`` string, optionally with alpha 0..255."""
    c = QColor(value)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def _channel(v: float) -> float:
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of ``#RRGGBB``."""
    c = QColor(hex_color)
    return (0.2126 * _channel(c.red()) + 0.7152 * _channel(c.green())
            + 0.0722 * _channel(c.blue()))


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio (1..21) between two ``#RRGGBB`` colours."""
    a, b = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def text_pairs() -> list[tuple[str, str, str, float]]:
    """Every (name, foreground, background, minimum ratio) text pairing the UI
    uses.  ``selftest_theme`` asserts each meets its minimum."""
    body, other = 7.0, 4.5
    return [
        ("body text on window",        TEXT,           BG,        body),
        ("body text on panel",         TEXT,           PANEL,     body),
        ("body text on inset",         TEXT,           INSET,     body),
        ("dim text on window",         TEXT_DIM,       BG,        body),
        ("dim text on panel",          TEXT_DIM,       PANEL,     body),
        ("dim text on inset",          TEXT_DIM,       INSET,     body),
        ("gold title on window",       GOLD[0],        BG,        body),
        ("gold title on panel",        GOLD[0],        PANEL,     body),
        ("gold readout on inset",      GOLD[0],        INSET,     body),
        ("green text on panel",        GREEN[0],       PANEL,     other),
        ("green text on inset",        GREEN[0],       INSET,     other),
        ("dark text on gold fill",     TEXT_ON_GOLD,   GOLD[0],   body),
        ("dark text on hover gold",    TEXT_ON_GOLD,   ACCENT_HI, body),
        ("white on purple fill",       TEXT_ON_PURPLE, PURPLE[0], body),
        ("white on purple (deep)",     TEXT_ON_PURPLE, PURPLE[2], body),
        ("white on green fill",        "#FFFFFF",      GREEN[2],  other),
        ("channel red on inset",       CHANNEL_R,      INSET,     other),
        ("channel blue on inset",      CHANNEL_B,      INSET,     other),
    ]


# ----------------------------------------------------------------------- fonts

_MONO_CANDIDATES = ("SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "Courier New")
_SANS_CANDIDATES = ("SF Pro Text", "Helvetica Neue", "Segoe UI", "Arial")
_family_cache: dict[tuple, str] = {}


def _first_installed(candidates: tuple[str, ...], fallback: str) -> str:
    """The first candidate font family actually installed on this machine.

    Listing families that don't exist makes Qt scan for aliases at startup and
    log a "missing font family" warning, so pick one that is really there.
    """
    if candidates not in _family_cache:
        installed = set(QFontDatabase.families())
        _family_cache[candidates] = next(
            (c for c in candidates if c in installed), fallback)
    return _family_cache[candidates]


def mono_family() -> str:
    """Installed monospace family used for numeric readouts."""
    return _first_installed(_MONO_CANDIDATES, "Courier")


def sans_family() -> str:
    """Installed UI sans-serif family used for labels."""
    return _first_installed(_SANS_CANDIDATES, "Helvetica")


def mono_font(size: int = 10, bold: bool = False) -> QFont:
    """Monospace font for numeric readouts."""
    f = QFont(mono_family())
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    f.setBold(bold)
    return f


def label_font(size: int = 10, bold: bool = False) -> QFont:
    """Clean UI sans for labels."""
    f = QFont(sans_family())
    f.setPointSize(size)
    f.setBold(bold)
    return f


def title_font(size: int = 9) -> QFont:
    """Small, bold, letter-spaced font for section titles (use uppercase text)."""
    f = label_font(size, bold=True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
    return f


# ------------------------------------------------------------------ stylesheet

def stylesheet() -> str:
    """The application stylesheet for all standard Qt widgets."""
    return f"""
* {{ color: {TEXT}; selection-background-color: {GOLD[2]}; selection-color: {TEXT_ON_GOLD}; }}
QMainWindow, QDialog, QWidget#central {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QMainWindow > QWidget {{ background: {BG}; }}

QToolTip {{ background: {INSET}; color: {TEXT}; border: 1px solid {GOLD[2]};
            padding: 4px 6px; }}

/* ---- panels & displays ---- */
QFrame[panel="true"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame[inset="true"] {{ background: {INSET}; border: 1px solid {BORDER}; border-radius: 4px; }}

/* ---- labels ---- */
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[role="title"] {{ color: {GOLD[0]}; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; }}
QLabel[role="value"] {{ color: {GOLD[0]}; font-family: "{mono_family()}"; }}

/* ---- buttons ---- */
QPushButton, QToolButton {{
    background: {PANEL_HI}; color: {TEXT}; border: 1px solid {BORDER_HI};
    border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {GOLD[0]}; color: {GOLD[0]}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {INSET}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {BORDER_HI}; border-color: {BORDER}; }}
QPushButton:focus {{ border-color: {GOLD[0]}; }}
QPushButton[variant="primary"] {{
    background: {GOLD[0]}; color: {TEXT_ON_GOLD}; border: 1px solid {GOLD[1]}; font-weight: 700; }}
QPushButton[variant="primary"]:hover {{ background: {ACCENT_HI}; color: {TEXT_ON_GOLD}; }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{
    background: transparent; border: 1px solid transparent; }}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{
    border-color: {GOLD[2]}; color: {GOLD[0]}; }}
QToolButton:checked {{ background: {GREEN[2]}; color: #FFFFFF; border-color: {GREEN[0]}; }}

/* ---- toolbar ---- */
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER};
            spacing: 4px; padding: 4px 8px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; padding: 5px 10px; }}
QToolBar QToolButton:hover {{ border-color: {GOLD[2]}; color: {GOLD[0]}; }}
QToolBar QToolButton:checked {{ background: {GREEN[2]}; color: #FFFFFF; border-color: {GREEN[0]}; }}

/* ---- combo box ---- */
QComboBox {{ background: {INSET}; color: {TEXT}; border: 1px solid {BORDER_HI};
             border-radius: 4px; padding: 3px 8px; }}
QComboBox:hover, QComboBox:focus {{ border-color: {GOLD[0]}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {INSET}; color: {TEXT}; border: 1px solid {GOLD[2]};
    selection-background-color: {GOLD[2]}; selection-color: {TEXT_ON_GOLD}; outline: none; }}

/* ---- lists ---- */
QListWidget {{ background: {INSET}; border: 1px solid {BORDER}; border-radius: 4px; outline: none; }}
QListWidget::item {{ color: {TEXT}; }}

/* ---- scroll bars (thin, dark) ---- */
QScrollBar:vertical {{ background: {INSET}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER_HI}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {GOLD[2]}; }}
QScrollBar:horizontal {{ background: {INSET}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER_HI}; border-radius: 4px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {GOLD[2]}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; width: 0; height: 0; }}

/* ---- status bar ---- */
QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}

/* ---- menus ---- */
QMenu {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 3px; }}
QMenu::item:selected {{ background: {GOLD[2]}; color: {TEXT_ON_GOLD}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

/* ---- dialogs ---- */
QMessageBox, QProgressDialog, QFileDialog {{ background: {BG}; }}
QProgressBar {{ background: {INSET}; border: 1px solid {BORDER}; border-radius: 4px;
                text-align: center; color: {TEXT}; }}
QProgressBar::chunk {{ background: {GOLD[0]}; border-radius: 3px; }}
QLineEdit {{ background: {INSET}; color: {TEXT}; border: 1px solid {BORDER_HI}; border-radius: 4px;
             padding: 3px 6px; }}
QLineEdit:focus {{ border-color: {GOLD[0]}; }}
QRadioButton, QCheckBox {{ color: {TEXT}; spacing: 6px; }}
"""


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, color(BG))
    p.setColor(QPalette.WindowText, color(TEXT))
    p.setColor(QPalette.Base, color(INSET))
    p.setColor(QPalette.AlternateBase, color(PANEL))
    p.setColor(QPalette.Text, color(TEXT))
    p.setColor(QPalette.Button, color(PANEL_HI))
    p.setColor(QPalette.ButtonText, color(TEXT))
    p.setColor(QPalette.BrightText, color("#FFFFFF"))
    p.setColor(QPalette.ToolTipBase, color(INSET))
    p.setColor(QPalette.ToolTipText, color(TEXT))
    p.setColor(QPalette.Highlight, color(GOLD[2]))
    p.setColor(QPalette.HighlightedText, color(TEXT_ON_GOLD))
    p.setColor(QPalette.PlaceholderText, color(TEXT_DIM))
    p.setColor(QPalette.Link, color(GOLD[0]))
    p.setColor(QPalette.Mid, color(BORDER))
    p.setColor(QPalette.Dark, color(INSET))
    p.setColor(QPalette.Light, color(PANEL_HI))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, color(BORDER_HI))
    return p


_APPLIED = "_blendstack_theme_applied"


def apply(app: QApplication | None = None) -> None:
    """Install the dark Fusion palette, default font and stylesheet.

    Idempotent; safe to call from both ``__main__`` and ``MainWindow``.
    Fusion is used because the native macOS style ignores most stylesheet and
    palette settings, which would make the theme inconsistent.
    """
    app = app or QApplication.instance()
    if app is None or app.property(_APPLIED):
        return
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setFont(label_font(11))
    app.setStyleSheet(stylesheet())
    app.setProperty(_APPLIED, True)
