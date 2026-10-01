"""Development launcher for the Rivelero GUI (without installing the package).

Installed users run ``rivelero`` or ``python -m rivelero`` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivelero.gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
