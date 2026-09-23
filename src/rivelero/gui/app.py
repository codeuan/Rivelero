"""Start the Rivelero desktop application.

Used by the ``rivelero`` command, ``python -m rivelero`` and the development
launcher ``tests/launch_new_gui.py``::

    rivelero                       # empty project
    rivelero "Sicily Survey.rivelero"   # open a saved project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WINDOW_SIZE = (1380, 860)


def build_parser() -> argparse.ArgumentParser:
    from rivelero import __version__

    parser = argparse.ArgumentParser(
        prog="rivelero",
        description="Rivelero: spatial observability of field surveys.",
    )
    parser.add_argument("project", nargs="?", type=Path,
                        help="a .rivelero project file to open")
    parser.add_argument("--version", action="version", version=f"Rivelero {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        try:
            from PyQt6.QtWidgets import QApplication
        except ModuleNotFoundError:
            print(
                "Rivelero's graphical interface needs PySide6. Install it with "
                "'pip install PySide6' (or the project environment.yml).",
                file=sys.stderr,
            )
            return 1
    except ImportError as error:
        # Installed but not loadable (e.g. mixed Qt libraries from conda and
        # pip): report the real cause instead of claiming it is missing.
        print(
            f"PySide6 is installed but could not be loaded: {error}\n"
            "This usually means Qt libraries from different installations are "
            "mixed (for example a pip PySide6 inside a conda environment). "
            "Install PySide6 from a single source.",
            file=sys.stderr,
        )
        return 1

    from rivelero.gui.application_state import ApplicationState
    from rivelero.gui.main_window import MainWindow
    from rivelero.gui.theme import apply_theme

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Rivelero")
    app.setOrganizationName("Rivelero")
    apply_theme(app)

    window = MainWindow(state=ApplicationState())
    window.resize(*WINDOW_SIZE)
    window.show()
    if arguments.project is not None:
        window.open_project(arguments.project)
    return app.exec()
