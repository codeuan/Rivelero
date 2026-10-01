"""OpenStreetMap background in the survey, world and observability maps."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from rivelero.core.configuration import ViewpointConfiguration
from rivelero.core.viewpoint import Viewpoint
from rivelero.gui.basemap import BASEMAP_LABEL
from rivelero.gui.raster_map import ELEVATION_ALPHA_OVER_BASEMAP, RasterMapWidget
from rivelero.gui.survey_map import SurveyMapWidget
from rivelero.visualization.basemap import OSM_ATTRIBUTION

from test_basemap import FakeFetcher

DEM = (
    Path(__file__).resolve().parents[1]
    / "rivelero_synthetic_observability" / "environments" / "dem_flat.tif"
)

app = QApplication.instance() or QApplication([])


def _offline(layer, fetcher):
    layer.fetch_tile = fetcher
    layer.synchronous = True


def _basemap_images(axes):
    return [image for image in axes.images if image.get_label() == BASEMAP_LABEL]


def _texts(axes):
    return [text.get_text() for text in axes.texts]


def _survey_widget():
    widget = SurveyMapWidget()
    widget.set_configuration(ViewpointConfiguration(
        configuration_id="osm",
        name="OSM",
        viewpoints=[
            Viewpoint(viewpoint_id="a", x=500100.0, y=4099100.0, crs="EPSG:32633"),
            Viewpoint(viewpoint_id="b", x=500900.0, y=4099800.0, crs="EPSG:32633"),
        ],
    ))
    return widget


def test_basemap_is_off_by_default_and_never_fetches():
    widget = _survey_widget()
    fetcher = FakeFetcher()
    _offline(widget.basemap, fetcher)
    widget.canvas.draw()

    assert not widget.basemap.toggle.isChecked()
    assert widget.basemap.toggle.isEnabled()
    assert _basemap_images(widget.ax) == []
    assert fetcher.calls == []


def test_survey_map_shows_basemap_without_changing_limits():
    widget = _survey_widget()
    _offline(widget.basemap, FakeFetcher())
    widget.canvas.draw()
    limits = (widget.ax.get_xlim(), widget.ax.get_ylim())

    widget.basemap.set_enabled(True)
    widget.basemap.update_now()
    widget.canvas.draw()

    (image,) = _basemap_images(widget.ax)
    assert image.get_zorder() < 0
    assert OSM_ATTRIBUTION in _texts(widget.ax)
    assert (widget.ax.get_xlim(), widget.ax.get_ylim()) == limits

    # Redrawing (which clears the axes) keeps the background.
    widget.set_selected_viewpoint_id("b")
    widget.canvas.draw()
    assert len(_basemap_images(widget.ax)) == 1

    # Reset fits the Viewpoints, not the background image.
    widget.ax.set_xlim(0, 1e7)
    widget._reset_view()
    assert widget.ax.get_xlim()[1] < 502000

    widget.basemap.set_enabled(False)
    assert _basemap_images(widget.ax) == []
    assert OSM_ATTRIBUTION not in _texts(widget.ax)


def test_basemap_failure_is_reported_on_the_map():
    widget = _survey_widget()
    _offline(widget.basemap, FakeFetcher(fail=True))
    widget.canvas.draw()

    widget.basemap.set_enabled(True)
    widget.basemap.update_now()

    assert "OpenStreetMap unavailable" in widget.basemap.message
    assert any("OpenStreetMap unavailable" in text for text in _texts(widget.ax))
    assert _basemap_images(widget.ax) == []


def test_basemap_needs_a_crs():
    widget = SurveyMapWidget()
    widget.set_configuration(ViewpointConfiguration(
        configuration_id="none", name="none", viewpoints=[],
    ))
    assert not widget.basemap.toggle.isEnabled()
    assert not widget.basemap.enabled


@pytest.mark.skipif(not DEM.is_file(), reason="synthetic DEM not available")
def test_raster_map_makes_elevation_translucent_over_basemap():
    widget = RasterMapWidget()
    _offline(widget.basemap, FakeFetcher())
    assert not widget.basemap.toggle.isEnabled()

    widget.set_raster(DEM)
    widget.canvas.draw()
    assert widget._raster_artist.get_alpha() is None

    widget.basemap.set_enabled(True)
    widget.basemap.update_now()
    widget.canvas.draw()

    assert widget._raster_artist.get_alpha() == ELEVATION_ALPHA_OVER_BASEMAP
    assert len(_basemap_images(widget.axes)) == 1
    assert widget.current_extent == widget.full_extent

    widget.clear_raster()
    assert not widget.basemap.enabled
    assert _basemap_images(widget.axes) == []
