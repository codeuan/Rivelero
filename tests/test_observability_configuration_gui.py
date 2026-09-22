"""O1 tests for the Observability visibility-configuration GUI."""

from pathlib import Path

import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import ElevationModel, Environment
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.application_state import ApplicationState
from rivelero.gui.observability_page import ObservabilityPage
from rivelero.visibility.configuration import MissingMetadataPolicy, SamplingUnit


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# Pages (which embed Matplotlib maps) stay referenced for the session so that
# queued redraws never reach a canvas whose C++ object was already deleted.
_PAGES = []


def _page(state):
    page = ObservabilityPage(state)
    _PAGES.append(page)
    return page


def _ready_state():
    state = ApplicationState()
    vp = Viewpoint(
        viewpoint_id="vp-1", x=500050.0, y=4099950.0,
        crs=CRS.from_epsg(32633),
    )
    state.set_viewpoint_configuration(ViewpointConfiguration(
        configuration_id="survey-1", name="Survey", viewpoints=[vp]
    ))
    grid = AnalysisGrid(
        crs=CRS.from_epsg(32633),
        transform=Affine(10, 0, 500000, 0, -10, 4100000),
        width=10, height=10,
    )
    env = Environment(
        environment_id="env-1", name="Terrain",
        elevation_model=ElevationModel(
            source=Path("terrain.tif"), crs=CRS.from_epsg(32633)
        ),
    )
    domain = AnalysisDomain(
        domain_id="domain-1", name="Domain",
        geometry=box(500000, 4099900, 500100, 4100000),
        crs=grid.crs, grid=grid,
    )
    state.set_environment(env, grid=grid)
    state.set_analysis_domain(domain)
    return state


def test_o1_saves_canonical_configuration(app):
    state = _ready_state()
    page = _page(state)
    page.max_distance.setValue(750.0)
    page._save_configuration()
    config = state.analysis.visibility_configuration
    assert config is not None
    assert config.max_distance_m == 750.0
    assert config.missing_heading_policy == MissingMetadataPolicy.OMNIDIRECTIONAL
    assert config.sampling_unit == SamplingUnit.VIEWPOINT
    assert not state.observability_ready


def test_o1_missing_source_metadata_is_not_modified(app):
    state = _ready_state()
    viewpoint = state.survey.viewpoint_configuration.viewpoints[0]
    assert viewpoint.heading_deg is None
    page = _page(state)
    page._save_configuration()
    assert viewpoint.heading_deg is None


def test_o1_rehydrates_existing_configuration(app):
    state = _ready_state()
    page = _page(state)
    page.max_distance.setValue(1234.0)
    page._save_configuration()
    page2 = _page(state)
    assert page2.max_distance.value() == pytest.approx(1234.0)
    assert page2.name_edit.text() == "Default visibility"


def test_event_sampling_requires_events(app):
    state = _ready_state()
    page = _page(state)
    page.sampling_unit.setCurrentIndex(
        page.sampling_unit.findData(SamplingUnit.OBSERVATION_EVENT)
    )
    with pytest.raises(ValueError, match="ObservationEvent"):
        page._configuration_from_form()


def test_height_policy_never_offers_omnidirectional(app):
    state = _ready_state()
    page = _page(state)
    values = [page.height_policy.itemData(i) for i in range(page.height_policy.count())]
    assert MissingMetadataPolicy.OMNIDIRECTIONAL not in values
