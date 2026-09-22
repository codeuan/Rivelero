from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_visibility_cache(tmp_path_factory):
    """Keep GUI-created VisibilityStores out of the real per-user cache."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv(
        "RIVELERO_CACHE_DIR",
        str(tmp_path_factory.mktemp("rivelero_cache")),
    )
    yield
    monkeypatch.undo()
