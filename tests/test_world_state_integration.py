from pathlib import Path

import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from rivelero.core.domain import AnalysisDomain, AnalysisGrid
from rivelero.core.environment import ElevationModel, Environment
from rivelero.gui.application_state import ApplicationState


def _grid():
    return AnalysisGrid(
        crs=CRS.from_epsg(32633),
        transform=Affine(10, 0, 500000, 0, -10, 4100000),
        width=10,
        height=10,
    )


def _environment():
    return Environment(
        environment_id="env-1",
        name="Terrain",
        elevation_model=ElevationModel(
            source=Path("terrain.tif"),
            crs=CRS.from_epsg(32633),
        ),
    )


def _domain(grid):
    return AnalysisDomain(
        domain_id="domain-1",
        name="Domain",
        geometry=box(500000, 4099900, 500100, 4100000),
        crs=grid.crs,
        grid=grid,
        analysis_mask=None,
    )


def test_world_state_requires_grid_and_domain_and_preserves_survey():
    state = ApplicationState()
    assert not state.world_ready
    grid = _grid()
    state.set_environment(_environment(), grid=grid)
    assert not state.world_ready
    state.set_analysis_domain(_domain(grid))
    assert state.world_ready
    state.set_analysis_domain(None)
    assert not state.world_ready


def test_world_state_rejects_domain_without_grid_or_mismatched_grid():
    state = ApplicationState()
    with pytest.raises(ValueError, match="AnalysisGrid"):
        state.set_analysis_domain(_domain(_grid()))
    state.set_environment(_environment(), grid=_grid())
    mismatched = _grid()
    mismatched.width = 9
    with pytest.raises(ValueError, match="does not match"):
        state.set_analysis_domain(_domain(mismatched))


def test_environment_replacement_clears_domain_and_rejects_missing_grid():
    state = ApplicationState()
    with pytest.raises(TypeError, match="AnalysisGrid"):
        state.set_environment(_environment())
    grid = _grid()
    state.set_environment(_environment(), grid=grid)
    state.set_analysis_domain(_domain(grid))
    state.set_environment(_environment(), grid=_grid())
    assert state.analysis.analysis_domain is None
    assert not state.world_ready
