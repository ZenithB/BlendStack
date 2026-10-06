"""Offscreen check of the theme: readability rules + that it installs cleanly.

    QT_QPA_PLATFORM=offscreen python -m blendstack.app.selftest_theme
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from blendstack.app import theme  # noqa: E402

_FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _FAILED
    if not ok:
        _FAILED += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    # Every text pairing the UI uses meets its readability minimum.
    for name, fg, bg, minimum in theme.text_pairs():
        ratio = theme.contrast_ratio(fg, bg)
        check(f"contrast: {name}", ratio >= minimum, f"{ratio:.1f}:1 (need {minimum}:1)")

    # The palette is exactly the one supplied.
    check("gold ramp", theme.GOLD == ["#FFC437", "#F3AC00", "#C48B00", "#A07100", "#745200"])
    check("purple ramp", theme.PURPLE == ["#792DAC", "#6509A3", "#510784", "#41046B", "#2F024E"])
    check("green ramp", theme.GREEN == ["#25AD7A", "#00A466", "#008553", "#006C43", "#004E31"])

    # Main background really is a dark grey (low, neutral luminance).
    c = theme.color(theme.BG)
    check("window background is dark grey",
          max(c.red(), c.green(), c.blue()) < 60 and
          max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue()) < 12)

    # Purple is only used as a fill: raw purple text would fail readability.
    check("purple is NOT usable as text on dark grey",
          theme.contrast_ratio(theme.PURPLE[0], theme.BG) < 4.5)

    theme.apply(app)
    theme.apply(app)  # idempotent
    # (Once a stylesheet is set Qt wraps the style in a proxy whose name is
    # empty, so check the observable effects instead of the style name.)
    check("theme applies (stylesheet + dark palette)",
          "QPushButton" in app.styleSheet()
          and app.palette().color(app.palette().ColorRole.Window).name().lower() == theme.BG.lower())
    print(f"\n{'ALL CHECKS PASSED' if not _FAILED else str(_FAILED) + ' FAILED'}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
