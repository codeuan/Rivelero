from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from rivelero.analysis.dem_distribution import (
    distribute_dem_coordinates,
    fill_dem_camera_metadata,
    generate_dem_viewpoints,
    select_dem_sites,
)


@pytest.fixture
def dem(tmp_path):
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=100, height=100, count=1,
        dtype="float32", crs="EPSG:32633", transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as raster:
        raster.write(np.ones((100, 100), dtype=np.float32), 1)
    return path


def test_lattice_uses_100_cell_centers_and_assigns_in_input_order(dem):
    inputs = [(x, y) for y in range(5, 100, 10) for x in range(5, 100, 10)]
    crs, positions = distribute_dem_coordinates(dem, inputs, mode="random")
    assert crs.to_epsg() == 32633
    assert positions == inputs
    assert set(select_dem_sites(dem, 100, mode="random")[1]) == set(inputs)


def test_modes_are_distinct_reproducible_and_use_unique_sites(dem):
    count = 7
    positions = {}
    for mode in ("random", "clustered", "dispersed"):
        result = select_dem_sites(dem, count, mode=mode, seed=12)
        assert result == select_dem_sites(dem, count, mode=mode, seed=12)
        positions[mode] = set(result[1])
        assert len(positions[mode]) == count

    def closest_pair(points):
        return min(np.hypot(x - u, y - v) for x, y in points for u, v in points
                   if (x, y) != (u, v))

    assert positions["random"] != positions["clustered"]
    assert positions["random"] != positions["dispersed"]
    assert closest_pair(positions["clustered"]) == 10
    assert closest_pair(positions["dispersed"]) >= 30


def test_nodata_sites_are_skipped_and_capacity_is_reported(dem):
    with rasterio.open(dem, "r+") as raster:
        values = raster.read(1)
        values[:10, :10] = -9999
        raster.write(values, 1)

    _, positions = select_dem_sites(dem, 99, mode="random")
    assert (5, 95) not in positions
    with pytest.raises(ValueError, match="Only 99"):
        select_dem_sites(dem, 100, mode="clustered")
    with pytest.raises(ValueError, match="finite"):
        distribute_dem_coordinates(dem, [(float("nan"), 0)], mode="random")


@pytest.mark.parametrize("mode", ["random", "clustered", "dispersed"])
def test_generated_viewpoints_keep_terrain_centre_headings_across_distributions(dem, mode):
    # Remove an edge site so that the centroid of the valid sites differs from
    # the DEM centre. Both batches must keep the same camera orientation.
    with rasterio.open(dem, "r+") as raster:
        raster.write(np.full((10, 10), -9999, dtype=np.float32), 1,
                     window=rasterio.windows.Window(0, 0, 10, 10))

    candidates = generate_dem_viewpoints(dem, 99, environment_id="terrain", mode="random")
    batch = generate_dem_viewpoints(
        dem, 20, environment_id="terrain", mode=mode, seed=12,
        existing_ids=(f"dem-{mode}-1",),
    )
    by_position = {(camera.x, camera.y): camera for camera in candidates}

    assert by_position[5, 5].heading_deg == pytest.approx(45)
    assert by_position[95, 5].heading_deg == pytest.approx(315)
    assert by_position[95, 95].heading_deg == pytest.approx(225)
    assert by_position[15, 95].heading_deg == pytest.approx(142.12501635)
    assert batch[0].viewpoint_id == f"dem-{mode}-2"
    for camera in batch:
        assert camera.horizontal_fov_deg == 120
        assert camera.heading_deg == by_position[camera.x, camera.y].heading_deg


def test_fill_dem_camera_metadata_preserves_existing_viewpoints(dem):
    camera, = generate_dem_viewpoints(dem, 1, environment_id="terrain", mode="random")
    legacy = replace(camera, heading_deg=None, horizontal_fov_deg=None)
    partial = replace(camera, horizontal_fov_deg=None, heading_deg=15)
    other_terrain = replace(legacy, provenance={"environment_id": "other"})
    imported = replace(legacy, source="Imported")

    filled, completed, skipped, untouched = fill_dem_camera_metadata(
        dem, "terrain", [legacy, partial, other_terrain, imported]
    )

    assert filled == camera
    assert completed.heading_deg == 15
    assert completed.horizontal_fov_deg == 120
    assert skipped is other_terrain
    assert untouched is imported
