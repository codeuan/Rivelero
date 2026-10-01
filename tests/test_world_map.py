"""Focused tests for the Rivelero World map."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from rasterio.crs import CRS

pytest.importorskip(
    "PySide6"
)

from PySide6.QtWidgets import QApplication

from rivelero.gui.domain_service import (
    build_domain_from_grid_extent,
)
from rivelero.gui.environment_import import (
    inspect_elevation_raster,
)
from rivelero.gui.world_map import (
    WorldMapWidget,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA\rivelero_synthetic_observability"
)

DEM = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)


@dataclass
class DummyViewpoint:
    viewpoint_id: str
    x: float
    y: float
    crs: CRS


@pytest.fixture(scope="module")
def app():

    application = (
        QApplication.instance()
        or QApplication([])
    )

    return application


def test_world_map_displays_viewpoints(
    app,
):

    widget = WorldMapWidget()

    widget.set_raster(
        DEM
    )

    widget.set_viewpoints(
        [
            DummyViewpoint(
                "vp1",
                500500.0,
                4099500.0,
                CRS.from_epsg(
                    32633
                ),
            ),
            DummyViewpoint(
                "vp2",
                500700.0,
                4099600.0,
                CRS.from_epsg(
                    32633
                ),
            ),
        ]
    )

    assert len(
        widget._resolved_viewpoints
    ) == 2


def test_world_map_accepts_canonical_domain(
    app,
):

    widget = WorldMapWidget()

    widget.set_raster(
        DEM
    )

    grid = inspect_elevation_raster(
        DEM
    ).analysis_grid

    domain = (
        build_domain_from_grid_extent(
            grid=grid,
            domain_id="complete",
            name="Complete",
        )
        .domain
    )

    widget.set_domain(
        domain
    )

    assert widget._domain is domain


def test_selected_viewpoint_can_be_zoomed(
    app,
):

    widget = WorldMapWidget()

    widget.set_raster(
        DEM
    )

    widget.set_viewpoints(
        [
            DummyViewpoint(
                "vp1",
                500500.0,
                4099500.0,
                CRS.from_epsg(
                    32633
                ),
            )
        ]
    )

    widget.set_selected_viewpoint(
        "vp1"
    )

    before = widget.full_extent

    widget.zoom_to_selected()

    after = widget.current_extent

    assert before is not None
    assert after is not None

    assert (
        after[1]
        - after[0]
    ) < (
        before[1]
        - before[0]
    )