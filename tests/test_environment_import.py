"""Tests for the World environment-import foundation."""

from pathlib import Path

from rasterio.crs import CRS

from rivelero.core.environment import (
    ElevationModelType,
)
from rivelero.gui.environment_import import (
    import_elevation_environment,
    inspect_elevation_raster,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA\rivelero_synthetic_observability"
)

DEM = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)


def test_inspect_synthetic_dem():

    metadata = inspect_elevation_raster(
        DEM
    )

    assert metadata.shape == (
        100,
        100,
    )

    assert metadata.crs == CRS.from_epsg(
        32633
    )

    assert metadata.width == 100
    assert metadata.height == 100

    assert metadata.resolution_x == 10.0
    assert metadata.resolution_y == 10.0
    assert metadata.resolution_m == 10.0

    assert metadata.bounds.left == 500000.0
    assert metadata.bounds.right == 501000.0

    assert metadata.bounds.bottom == 4099000.0
    assert metadata.bounds.top == 4100000.0


def test_import_environment_from_synthetic_dem():

    result = import_elevation_environment(
        DEM,
        environment_id="synthetic_environment",
        name="Synthetic flat environment",
        model_type=ElevationModelType.DEM,
        source_name="Rivelero synthetic data",
    )

    environment = result.environment

    assert (
        environment.environment_id
        == "synthetic_environment"
    )

    assert (
        environment.elevation_model.model_type
        == ElevationModelType.DEM
    )

    assert (
        environment.elevation_model.crs
        == CRS.from_epsg(32633)
    )

    assert (
        environment.elevation_model.resolution_m
        == 10.0
    )

    assert result.grid.shape == (
        100,
        100,
    )

    assert result.grid.crs == CRS.from_epsg(
        32633
    )

    assert result.grid.resolution_x == 10.0
    assert result.grid.resolution_y == 10.0


def test_dsm_is_representable_but_not_yet_gui_enabled():

    result = import_elevation_environment(
        DEM,
        environment_id="synthetic_dsm",
        name="Synthetic DSM representation",
        model_type=ElevationModelType.DSM,
    )

    assert (
        result.environment.is_surface_model
    )