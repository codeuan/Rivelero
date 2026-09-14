import os
import math
import tempfile
from pathlib import Path
from typing import Sequence, Mapping, Any

import requests


OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")

def _bbox_from_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    """
    Compute a WGS84 bounding box around lon/lat points with a buffer in metres.

    Args:
        sample_metadata:
            A sequence of dicts, each containing "lon" and "lat" keys.
        buffer_m:
            Buffer distance in metres to add around the points.

    Returns:
        (south, north, west, east)
    """
    if not sample_metadata:
        raise ValueError("sample_metadata is empty.")

    lons = [float(s["lon"]) for s in sample_metadata]
    lats = [float(s["lat"]) for s in sample_metadata]

    center_lat = sum(lats) / len(lats)
    lat_buffer_deg = buffer_m / 111_320.0
    lon_buffer_deg = buffer_m / (111_320.0 * max(0.1, math.cos(math.radians(center_lat))))

    south = min(lats) - lat_buffer_deg
    north = max(lats) + lat_buffer_deg
    west = min(lons) - lon_buffer_deg
    east = max(lons) + lon_buffer_deg

    return south, north, west, east


def _download_binary_to_tempfile(content: bytes, suffix: str) -> str:
    """
    Write binary content to a temporary file and return the path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


# -------------------------------------------------------------------
# OpenTopography DEM functions
# -------------------------------------------------------------------

def download_dem_from_opentopo(
    south: float,
    north: float,
    west: float,
    east: float,
    demtype: str = "COP30",
) -> str:
    """
    Download a DEM GeoTIFF from OpenTopography and return the local file path.
    """
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY is not set.")

    url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        raise RuntimeError(
            "OpenTopography returned HTML instead of a GeoTIFF. "
            "Check your parameters and API key."
        )

    return _download_binary_to_tempfile(response.content, suffix=".tif")


def download_dem_for_samples(
    sample_metadata: Sequence[Mapping[str, Any]],
    max_distance_m: float,
    demtype: str = "COP30",
) -> str:
    """
    Compute a bounding box around sample points and download a DEM for that area.
    """
    south, north, west, east = _bbox_from_samples(sample_metadata, max_distance_m)
    return download_dem_from_opentopo(
        south=south,
        north=north,
        west=west,
        east=east,
        demtype=demtype,
    )

