"""Tests for Rivelero World spatial QC."""

from dataclasses import dataclass
from pathlib import Path

from rasterio.crs import CRS

from rivelero.gui.domain_service import (
    build_domain_from_grid_extent,
)
from rivelero.gui.environment_import import (
    inspect_elevation_raster,
)
from rivelero.gui.world_qc import (
    QCSeverity,
    run_world_qc,
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


def _grid():
    return inspect_elevation_raster(
        DEM
    ).analysis_grid


def test_all_viewpoints_inside_terrain():

    grid = _grid()

    viewpoints = [
        DummyViewpoint(
            "vp1",
            500500.0,
            4099500.0,
            CRS.from_epsg(32633),
        ),
        DummyViewpoint(
            "vp2",
            500600.0,
            4099600.0,
            CRS.from_epsg(32633),
        ),
    ]

    report = run_world_qc(
        viewpoints=viewpoints,
        grid=grid,
    )

    assert report.compatible

    assert report.viewpoints_inside_terrain == 2

    assert report.viewpoints_outside_terrain == 0


def test_outside_viewpoint_is_warning_not_coordinate_error():

    grid = _grid()

    viewpoints = [
        DummyViewpoint(
            "inside",
            500500.0,
            4099500.0,
            CRS.from_epsg(32633),
        ),
        DummyViewpoint(
            "outside",
            600000.0,
            4200000.0,
            CRS.from_epsg(32633),
        ),
    ]

    report = run_world_qc(
        viewpoints=viewpoints,
        grid=grid,
    )

    assert report.viewpoints_inside_terrain == 1
    assert report.viewpoints_outside_terrain == 1

    issue = next(
        issue
        for issue in report.issues
        if issue.code
        == "viewpoints_outside_terrain"
    )

    assert issue.severity == (
        QCSeverity.WARNING
    )

    assert issue.viewpoint_ids == (
        "outside",
    )


def test_domain_and_viewpoint_counts():

    grid = _grid()

    domain = build_domain_from_grid_extent(
        grid=grid,
        domain_id="complete",
        name="Complete",
    ).domain

    viewpoints = [
        DummyViewpoint(
            "inside",
            500500.0,
            4099500.0,
            CRS.from_epsg(32633),
        ),
        DummyViewpoint(
            "outside",
            600000.0,
            4200000.0,
            CRS.from_epsg(32633),
        ),
    ]

    report = run_world_qc(
        viewpoints=viewpoints,
        grid=grid,
        domain=domain,
    )

    assert report.domain_overlaps_terrain
    assert report.domain_fully_inside_terrain

    assert report.viewpoints_inside_domain == 1
    assert report.viewpoints_outside_domain == 1


def test_different_crs_is_explicitly_transformed():

    grid = _grid()

    # Approximately the WGS84 position corresponding to the UTM synthetic
    # environment. Exact equality is not required; this tests the explicit
    # reprojection path.
    viewpoints = [
        DummyViewpoint(
            "wgs84",
            15.0,
            37.04,
            CRS.from_epsg(4326),
        )
    ]

    report = run_world_qc(
        viewpoints=viewpoints,
        grid=grid,
    )

    assert not report.has_errors

    assert any(
        issue.code
        == "survey_terrain_crs_differ"
        for issue in report.issues
    )


def test_empty_survey_is_nonblocking_warning():

    grid = _grid()

    report = run_world_qc(
        viewpoints=[],
        grid=grid,
    )

    assert report.compatible

    assert report.total_viewpoints == 0

    assert any(
        issue.code
        == "survey_empty"
        for issue in report.issues
    )