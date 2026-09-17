"""Development launcher for the new Rivelero GUI."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    from PyQt6.QtWidgets import QApplication

from rivelero.gui.application_state import ApplicationState
from rivelero.gui.main_window import MainWindow
from rivelero.gui.theme import apply_theme


def main() -> int:

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Rivelero"
    )
    apply_theme(
        app
    )

    state = ApplicationState()

    window = MainWindow(
        state=state
    )

    window.resize(
        1380,
        860,
    )

    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )