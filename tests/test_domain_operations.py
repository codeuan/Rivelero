"""Tests for Rivelero World AnalysisDomain construction."""

from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS

from rivelero.gui.domain_service import (
    DomainConstructionError,
    build_domain_from_drawn_polygon,
    build_domain_from_geojson,
    build_domain_from_grid_extent,
    build_domain_from_survey_extent,
)
from rivelero.gui.environment_import import (
    inspect_elevation_raster,
)
from rivelero.gui.survey_import import (
    SurveyImportOptions,
    import_survey_csv,
)


DATA_ROOT = Path(
    r"C:\Users\zool2620\VISTA\rivelero_synthetic_observability"
)

DEM = (
    DATA_ROOT
    / "environments"
    / "dem_flat.tif"
)

DOMAIN_GEOJSON = (
    DATA_ROOT
    / "domains"
    / "standard_domain.geojson"
)

VIEWPOINTS = (
    DATA_ROOT
    / "viewpoints"
    / "viewpoints.csv"
)

SENSORS = (
    DATA_ROOT
    / "viewpoints"
    / "sensors.csv"
)

EVENTS = (
    DATA_ROOT
    / "viewpoints"
    / "observation_events.csv"
)


def _grid():
    return inspect_elevation_raster(
        DEM
    ).analysis_grid


def _viewpoints():

    result = import_survey_csv(
        viewpoints_path=VIEWPOINTS,
        sensors_path=SENSORS,
        observation_events_path=EVENTS,
        configuration_id="world_test",
        configuration_name="World test",
        options=SurveyImportOptions(
            source_crs="EPSG:32633",
            target_crs="EPSG:32633",
            strict=True,
        ),
    )

    return list(
        result
        .viewpoint_configuration
        .viewpoints
    )


def test_complete_grid_domain():

    grid = _grid()

    result = build_domain_from_grid_extent(
        grid=grid,
        domain_id="complete",
        name="Complete terrain",
    )

    domain = result.domain

    assert domain.crs == CRS.from_epsg(
        32633
    )

    assert domain.shape == (
        100,
        100,
    )

    assert np.all(
        domain.analysis_mask
    )

    assert domain.creation_method == (
        "environment_extent"
    )


def test_drawn_polygon_domain():

    grid = _grid()

    result = build_domain_from_drawn_polygon(
        vertices=[
            (
                500200.0,
                4099200.0,
            ),
            (
                500800.0,
                4099200.0,
            ),
            (
                500800.0,
                4099800.0,
            ),
            (
                500200.0,
                4099800.0,
            ),
        ],
        map_crs="EPSG:32633",
        grid=grid,
        domain_id="drawn",
        name="Drawn domain",
    )

    assert np.any(
        result.domain.analysis_mask
    )

    assert not np.all(
        result.domain.analysis_mask
    )

    assert result.domain.creation_method == (
        "user_drawn_polygon"
    )


def test_import_synthetic_domain():

    grid = _grid()

    result = build_domain_from_geojson(
        path=DOMAIN_GEOJSON,
        source_crs="EPSG:32633",
        grid=grid,
        domain_id="imported",
        name="Imported domain",
        role="analysis_domain",
    )

    assert result.domain.geometry.area == pytest.approx(
        1_000_000.0
    )

    assert np.all(
        result.domain.analysis_mask
    )


def test_buffered_survey_domain():

    grid = _grid()

    result = build_domain_from_survey_extent(
        viewpoints=_viewpoints(),
        grid=grid,
        domain_id="survey_buffer",
        name="Survey buffer",
        buffer_m=50.0,
    )

    assert result.domain.geometry.area > 0

    assert np.any(
        result.domain.analysis_mask
    )

    assert result.domain.creation_method == (
        "buffered_survey_extent"
    )


def test_zero_buffer_single_point_rejected():

    grid = _grid()

    viewpoint = _viewpoints()[0]

    with pytest.raises(
        DomainConstructionError
    ):
        build_domain_from_survey_extent(
            viewpoints=[
                viewpoint
            ],
            grid=grid,
            domain_id="zero",
            name="Zero-area domain",
            buffer_m=0.0,
        )


def test_outside_drawn_polygon_rejected():

    grid = _grid()

    with pytest.raises(
        DomainConstructionError
    ):
        build_domain_from_drawn_polygon(
            vertices=[
                (
                    600000.0,
                    4200000.0,
                ),
                (
                    600100.0,
                    4200000.0,
                ),
                (
                    600100.0,
                    4200100.0,
                ),
            ],
            map_crs="EPSG:32633",
            grid=grid,
            domain_id="outside",
            name="Outside",
        )