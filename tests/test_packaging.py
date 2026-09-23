"""P4 tests: package version, entry points and headless imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(code: str | None = None, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    env.pop("QT_QPA_PLATFORM", None)
    command = [sys.executable, *(("-c", code) if code else ()), *args]
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)


def test_version_is_single_sourced():
    import rivelero
    from rivelero.project.schema import software_metadata

    assert rivelero.__version__ == "0.1.0.dev0"
    assert software_metadata()["version"] == rivelero.__version__
    text = (SRC.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = { attr = "rivelero.__version__" }' in text
    assert 'rivelero = "rivelero.gui.app:main"' in text


def test_scientific_imports_are_headless():
    """Scientific modules never import Qt (no display server needed)."""
    result = _run(
        "import sys\n"
        "import rivelero, rivelero.observability, rivelero.analysis.coverage\n"
        "import rivelero.analysis.comparison, rivelero.project.io\n"
        "import rivelero.export.catalog, rivelero.export.provenance\n"
        "import rivelero.export.figures, rivelero.export.report\n"
        "qt = sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'PyQt6', 'shiboken6'))\n"
        "gui = sorted(m for m in sys.modules if m.startswith('rivelero.gui'))\n"
        "print(qt, gui)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[] []"


def test_import_rivelero_is_lightweight():
    result = _run(
        "import sys; import rivelero\n"
        "print(sorted(m for m in sys.modules if m.startswith('rivelero')))\n"
        "print('numpy' in sys.modules, 'matplotlib' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["['rivelero']", "False False"]


def test_module_entry_point_version_without_qt():
    result = _run(None, "-m", "rivelero", "--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Rivelero 0.1.0.dev0"


def test_entry_point_parses_project_argument():
    from rivelero.gui.app import build_parser

    arguments = build_parser().parse_args(["Sicily Survey.rivelero"])
    assert arguments.project == Path("Sicily Survey.rivelero")


def test_matplotlib_uses_the_same_qt_binding_as_rivelero():
    """Importing a Rivelero canvas first must not let Matplotlib pick PyQt6."""
    result = _run(
        "import sys\n"
        "import rivelero.gui.canvas\n"
        "import matplotlib.backends.qt_compat as qt\n"
        "import rivelero.gui.main_window\n"
        "print(qt.QT_API, 'PyQt6.QtWidgets' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PySide6 False"
