"""OpenStreetMap obstacle geometry acquisition.

This module is responsible only for obtaining building/barrier/etc. geometries
from OpenStreetMap and returning them in a requested projected CRS.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

try:
    import osmnx as ox
except ImportError as error:
    ox = None
    _OSMNX_IMPORT_ERROR = error
else:
    _OSMNX_IMPORT_ERROR = None

DEFAULT_OBSTACLE_TAGS: dict[str, Any] = {
    "building": True,
    "barrier": True,
    "bridge": True,
    "tunnel": True,
    "man_made": [
        "tower", "mast", "chimney", "silo", "storage_tank",
        "water_tower", "crane", "antenna", "bridge", "tunnel",
    ],
    "power": ["tower", "pole"],
}

__all__ = ["DEFAULT_OBSTACLE_TAGS", "fetch_obstacles_for_extent", "projected_bounds_to_wgs84"]

def projected_bounds_to_wgs84(*, left: float, right: float, bottom: float, top: float, projected_crs: CRS | str) -> tuple[float, float, float, float]:
    """Convert projected bounds to ``(left, bottom, right, top)`` in EPSG:4326."""
    return tuple(float(v) for v in transform_bounds(projected_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21))

def fetch_obstacles_for_extent(*, left: float, right: float, bottom: float, top: float, projected_crs: CRS | str, tags: dict[str, Any] | None = None, requests_timeout: int = 60, use_cache: bool = True) -> gpd.GeoDataFrame:
    """Query OSM obstacle-like features and reproject them to ``projected_crs``."""
    if ox is None:
        raise ImportError("OSMnx is required to download obstacle features. Install osmnx or supply a pre-fetched GeoDataFrame to obstacles.py.") from _OSMNX_IMPORT_ERROR
    tags = tags or DEFAULT_OBSTACLE_TAGS
    ox.settings.requests_timeout = int(requests_timeout)
    ox.settings.use_cache = bool(use_cache)
    ox.settings.log_console = False
    bbox_wgs84 = projected_bounds_to_wgs84(left=left, right=right, bottom=bottom, top=top, projected_crs=projected_crs)
    try:
        raw = ox.features.features_from_bbox(bbox=bbox_wgs84, tags=tags)
    except Exception as error:
        if error.__class__.__name__ in {"InsufficientResponseError", "EmptyOverpassResponse"}:
            return _empty_geodataframe(projected_crs)
        raise
    if raw.empty:
        return _empty_geodataframe(projected_crs)
    raw = raw.reset_index(drop=False)
    if raw.crs is None:
        raw = raw.set_crs("EPSG:4326")
    return raw.to_crs(projected_crs)

def _empty_geodataframe(crs: CRS | str | None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs=crs)}, geometry="geometry", crs=crs)



