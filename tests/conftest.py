import os
from pathlib import Path
import sys

# GUI tests never open real windows (and run on headless machines). Set
# QT_QPA_PLATFORM explicitly to watch them on screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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


# ---------------------------------------------------------------------------
# Test categories (markers are registered in pyproject.toml)
# ---------------------------------------------------------------------------
#
#   unit         Qt-free and fast: models, analysis arithmetic, export writers.
#   integration  Runs real viewsheds / SOF builds or real file round-trips.
#   gui          Needs Qt widgets (runs offscreen with QT_QPA_PLATFORM=offscreen).
#   slow         The files dominating the full-suite time (GUI workflows that
#                build fields, run tasks and render reports).
#
# Fast development loop:  pytest -m "not slow"
# Authoritative check:    pytest
#
# Every test module must be listed here, so new modules are classified
# deliberately rather than silently falling into the fast set.

TEST_CATEGORIES: dict[str, tuple[str, ...]] = {
    "test_analysis_comparison.py": ("unit",),
    "test_analysis_contribution.py": ("integration",),
    "test_analysis_coverage.py": ("integration",),
    "test_analysis_page.py": ("gui", "integration"),
    "test_analysis_scenario.py": ("unit",),
    "test_compare_panel.py": ("gui", "integration", "slow"),
    "test_contribution_panel.py": ("gui", "integration", "slow"),
    "test_dem_io.py": ("unit",),
    "test_domain_operations.py": ("unit",),
    "test_environment_import.py": ("unit",),
    "test_export.py": ("unit",),
    "test_figures.py": ("unit",),
    "test_lifecycle.py": ("gui", "integration", "slow"),
    "test_observability_build.py": ("gui", "integration"),
    "test_observability_configuration_gui.py": ("gui",),
    "test_observability_integration.py": ("gui", "integration", "slow"),
    "test_observability_map.py": ("gui", "integration"),
    "test_output_page.py": ("gui", "integration", "slow"),
    "test_packaging.py": ("unit",),
    "test_project_gui.py": ("gui", "integration", "slow"),
    "test_project_persistence.py": ("integration", "slow"),
    "test_provenance_report.py": ("integration", "slow"),
    "test_raster_map.py": ("gui",),
    "test_scenario_panel.py": ("gui", "integration", "slow"),
    "test_schema_compatibility.py": ("integration",),
    "test_survey_dialogs.py": ("gui",),
    "test_survey_import.py": ("unit",),
    "test_survey_map.py": ("gui",),
    "test_survey_operations.py": ("unit",),
    "test_synthetic_flat.py": ("integration",),
    "test_synthetic_observability.py": ("integration",),
    "test_world_map.py": ("gui",),
    "test_world_qc.py": ("unit",),
    "test_world_state_integration.py": ("unit",),
}


def pytest_collection_modifyitems(config, items):
    unclassified = set()
    for item in items:
        name = item.path.name
        categories = TEST_CATEGORIES.get(name)
        if categories is None:
            unclassified.add(name)
            continue
        for category in categories:
            item.add_marker(getattr(pytest.mark, category))
    if unclassified:
        raise pytest.UsageError(
            "Unclassified test module(s): " + ", ".join(sorted(unclassified))
            + ". Add them to TEST_CATEGORIES in tests/conftest.py."
        )
