"""Focused tests for reusable Rivelero raster-map infrastructure."""

from pathlib import Path

import pytest

pytest.importorskip(
    "PySide6"
)

from PySide6.QtWidgets import QApplication

from rivelero.gui.raster_map import (
    RasterMapWidget,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA\rivelero_synthetic_observability"
)

DEM = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)


@pytest.fixture(scope="module")
def app():

    application = (
        QApplication.instance()
        or QApplication([])
    )

    return application


def test_raster_map_loads_synthetic_dem(
    app,
):

    widget = RasterMapWidget()

    widget.set_raster(
        DEM
    )

    assert widget.metadata is not None

    assert widget.metadata.shape == (
        100,
        100,
    )

    assert widget.full_extent == (
        500000.0,
        501000.0,
        4099000.0,
        4100000.0,
    )


def test_raster_map_extent_can_change(
    app,
):

    widget = RasterMapWidget()

    widget.set_raster(
        DEM
    )

    widget.set_view_extent(
        (
            500200.0,
            500800.0,
            4099200.0,
            4099800.0,
        ),
        reload=False,
    )

    extent = widget.current_extent

    assert extent is not None

    assert extent[0] == pytest.approx(
        500200.0
    )

    assert extent[1] == pytest.approx(
        500800.0
    )