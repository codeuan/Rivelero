#obstacle_detection.py
#Query nearby OpenStreetMap obstacle-like features, convert them into the DEM CRS,
#Build a rectangular bounding box around each one to be displayed on the overlay.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import geopandas as gpd
import osmnx as ox
from matplotlib.axes import Axes
from pyproj import CRS, Transformer
from shapely.geometry import box

DEFAULT_OBSTACLE_TAGS: dict[str, Any] = {
    "building": True,
    "barrier": True,
    "bridge": True,
    "tunnel": True,
    "man_made": [
        "tower",
        "mast",
        "chimney",
        "silo",
        "storage_tank",
        "water_tower",
        "crane",
        "antenna",
        "bridge",
        "tunnel"
    ],
    "power": ["tower", "pole"],
} #obstacle filter for API call.


@dataclass(slots=True)
class ObstacleResult: #class for return value from API call.
    raw_features: gpd.GeoDataFrame
    bbox_features: gpd.GeoDataFrame
    query_bbox_wgs84: tuple[float, float, float, float]  # left, bottom, right, top


def projected_bounds_to_wgs84(
    left: float,
    right: float,
    bottom: float,
    top: float,
    projected_crs: CRS | str,
) -> tuple[float, float, float, float]:
    """
    Convert a projected map extent into an EPSG:4326 bbox for OSMnx.

    Returns:
        (left, bottom, right, top) in lon/lat degrees.
    """
    transformer = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True) #coordinate transformer.

   
    corners_xy = [
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
    ] #grab 4 corners of current region being analysed.
    corners_lonlat = [transformer.transform(x, y) for x, y in corners_xy] #convert x,y into lon,lat.

    lons = [lon for lon, _ in corners_lonlat] #only the longitudes.
    lats = [lat for _, lat in corners_lonlat] #only the latitudes.

    return min(lons), min(lats), max(lons), max(lats) #bounding box for region being analysed.


def fetch_obstacles_for_extent(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    projected_crs: CRS | str,
    tags: dict[str, Any] | None = None,
    requests_timeout: int = 60,
    use_cache: bool = True,
) -> ObstacleResult:
    """
    Query OpenStreetMap/Overpass for nearby obstacle-like features and compute
    a rectangular bounding box for each one.

    Returns GeoDataFrames already reprojected into `projected_crs`, so they can
    be drawn directly on the DEM axes.
    """
    tags = tags or DEFAULT_OBSTACLE_TAGS #predefined tags for what constitutes an obstacle.

    ox.settings.requests_timeout = requests_timeout #defines timeout.
    ox.settings.use_cache = use_cache #use cached results when possible rather than making further web requests.
    ox.settings.log_console = False #no console logging from OSMnx as it is unnecessary.

    bbox_wgs84 = projected_bounds_to_wgs84(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        projected_crs=projected_crs,
    ) #format for search area in OSMnx.

    raw_gdf = ox.features.features_from_bbox(bbox=bbox_wgs84, tags=tags) #query features in bounding box.

    if raw_gdf.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=projected_crs) #if nothing returned, make an empty GeoDataFrame.
        return ObstacleResult(
            raw_features=empty,
            bbox_features=empty,
            query_bbox_wgs84=bbox_wgs84,
        ) #return obstacle results with empty data.

    raw_gdf = raw_gdf.reset_index(drop=False)

    if raw_gdf.crs is None:
        raw_gdf = raw_gdf.set_crs("EPSG:4326")

    raw_gdf = raw_gdf.to_crs(projected_crs) #raw outlines of obstacles.


    bounds_df = raw_gdf.geometry.bounds #rectangle around obstacle boundaries.
    bbox_geoms = [
        box(minx, miny, maxx, maxy)
        for minx, miny, maxx, maxy
        in bounds_df[["minx", "miny", "maxx", "maxy"]].itertuples(index=False, name=None)
    ] #build a Shapely rectangle.

    bbox_gdf = raw_gdf.copy() #GeoDataFrame identical to raw_gdf.
    bbox_gdf["geometry"] = bbox_geoms #rectangle geometry from bounding box.

    return ObstacleResult(
        raw_features=raw_gdf,
        bbox_features=bbox_gdf,
        query_bbox_wgs84=bbox_wgs84,
    )


def draw_obstacle_bboxes(
    ax: Axes,
    bbox_gdf: gpd.GeoDataFrame | None,
    *,
    linewidth: float = 1.0,
    linestyle: str = "--",
    edgecolor: str = "cyan",
    alpha: float = 0.9,
    zorder: int = 6,
) -> None:
    """
    Draw each obstacle bounding box as an outline on an existing Matplotlib axes.
    """
    if bbox_gdf is None or bbox_gdf.empty:
        return #if no data was returned, stop.

    for geom in bbox_gdf.geometry: #for each object.
        if geom is None or geom.is_empty:
            continue #if geometry is missing, stop.

        boundary = geom.boundary #outline of eometry.

        if boundary.geom_type == "LineString": #if a single line makes up the obstacle.
            x, y = boundary.xy
            ax.plot(
                x,
                y,
                linewidth=linewidth,
                linestyle=linestyle,
                color=edgecolor,
                alpha=alpha,
                zorder=zorder,
            ) #draw said line.
        else:
            for part in getattr(boundary, "geoms", []):
                x, y = part.xy
                ax.plot(
                    x,
                    y,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=edgecolor,
                    alpha=alpha,
                    zorder=zorder,
                ) #for each part of the object's boundary, plot it.
