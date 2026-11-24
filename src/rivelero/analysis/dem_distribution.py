"""Place simulated viewpoints on a regular lattice over an elevation raster."""

from collections.abc import Iterable
from math import atan2, degrees
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import xy

from rivelero.core.viewpoint import Viewpoint

DistributionMode = Literal["random", "clustered", "dispersed"]
DISTRIBUTION_MODES = ("random", "clustered", "dispersed")


def distribute_dem_coordinates(
    source: Path | str,
    coordinates: list[tuple[float, float]],
    *,
    mode: DistributionMode,
    seed: int = 0,
) -> tuple[CRS, list[tuple[float, float]]]:
    """Assign input XY pairs to distinct sites, in input order.

    Sites are the centers of 10 equal subdivisions along each raster axis.
    Masked or nonfinite elevations are ineligible. Distances use the DEM's
    native coordinate units. Each input takes its nearest remaining site from
    the subset selected by the distribution mode.
    """
    if not 1 <= len(coordinates) <= 100:
        raise ValueError("Enter between 1 and 100 coordinate pairs.")
    inputs = np.asarray(coordinates, dtype=float)
    if inputs.shape != (len(coordinates), 2) or not np.isfinite(inputs).all():
        raise ValueError("Each coordinate pair must contain two finite numbers.")
    crs, positions = select_dem_sites(source, len(inputs), mode=mode, seed=seed)
    available = np.asarray(positions)
    assigned = []
    for x, y in inputs:
        nearest_index = int(np.argmin(np.hypot(available[:, 0] - x, available[:, 1] - y)))
        site_x, site_y = available[nearest_index]
        assigned.append((float(site_x), float(site_y)))
        available = np.delete(available, nearest_index, axis=0)
    return crs, assigned


def select_dem_sites(
    source: Path | str,
    count: int,
    *,
    mode: DistributionMode,
    seed: int = 0,
) -> tuple[CRS, list[tuple[float, float]]]:
    """Select distinct valid centers from a 10 × 10 grid over the DEM."""
    crs, positions, _ = _select_dem_sites(source, count, mode=mode, seed=seed)
    return crs, positions


def generate_dem_viewpoints(
    source: Path | str,
    count: int,
    *,
    environment_id: str,
    mode: DistributionMode,
    seed: int = 0,
    existing_ids: Iterable[str] = (),
) -> list[Viewpoint]:
    """Generate 120-degree cameras aimed at the centre of the DEM."""
    crs, positions, (centre_x, centre_y) = _select_dem_sites(
        source, count, mode=mode, seed=seed
    )
    occupied_ids = set(existing_ids)
    viewpoints = []
    number = 0
    for x, y in positions:
        number += 1
        while f"dem-{mode}-{number}" in occupied_ids:
            number += 1
        heading = degrees(atan2(centre_x - x, centre_y - y)) % 360.0
        viewpoints.append(Viewpoint(
            viewpoint_id=f"dem-{mode}-{number}",
            x=x,
            y=y,
            crs=crs,
            heading_deg=heading,
            horizontal_fov_deg=120.0,
            source="DEM lattice simulation",
            provenance={
                "environment_id": environment_id,
                "grid_shape": [10, 10],
                "distribution": mode,
                "seed": seed,
            },
        ))
    return viewpoints


def _select_dem_sites(
    source: Path | str,
    count: int,
    *,
    mode: DistributionMode,
    seed: int,
) -> tuple[CRS, list[tuple[float, float]], tuple[float, float]]:
    if not 1 <= count <= 100:
        raise ValueError("Request between 1 and 100 viewpoints.")
    if mode not in DISTRIBUTION_MODES:
        raise ValueError(f"Unknown distribution mode: {mode!r}.")

    with rasterio.open(source) as dem:
        crs = dem.crs
        if crs is None:
            raise ValueError("The DEM must have a coordinate reference system.")
        centre = xy(dem.transform, dem.height / 2, dem.width / 2, offset="ul")
        centers = np.arange(10) + 0.5
        rows = np.repeat(centers, 10) * dem.height / 10
        columns = np.tile(centers, 10) * dem.width / 10
        xs, ys = xy(dem.transform, rows, columns, offset="ul")
        lattice = list(zip(xs, ys, strict=True))
        sites = np.array([
            point
            for point, (height,) in zip(
                lattice, dem.sample(lattice, indexes=1, masked=True), strict=True
            )
            if not np.ma.is_masked(height) and np.isfinite(height)
        ])

    if count > len(sites):
        raise ValueError(
            f"Only {len(sites)} of the 100 lattice sites have valid elevation. "
            f"Request at most {len(sites)} viewpoints or use another DEM."
        )

    rng = np.random.default_rng(seed)
    if mode == "random":
        selected = rng.choice(len(sites), size=count, replace=False)
    else:
        anchor = int(rng.integers(len(sites)))
        distances = np.hypot(
            sites[:, None, 0] - sites[:, 0],
            sites[:, None, 1] - sites[:, 1],
        )
        if mode == "clustered":
            selected = np.argsort(distances[anchor], kind="stable")[:count]
        else:
            selected = [anchor]
            nearest = distances[anchor].copy()
            while len(selected) < count:
                nearest[selected] = -np.inf
                next_site = int(np.argmax(nearest))
                selected.append(next_site)
                nearest = np.minimum(nearest, distances[next_site])

    return crs, [(float(x), float(y)) for x, y in sites[selected]], centre
