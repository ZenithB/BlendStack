"""``python -m blendstack.app`` — QApplication bootstrap (brief §5)."""

from __future__ import annotations

import sys


def main() -> int:
    """Create the QApplication and run the main window's event loop."""
    from PySide6.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("BlendStack")
    app.setOrganizationName("BlendStack")
    theme.apply(app)  # dark "synth rack" look, before any widget exists
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
