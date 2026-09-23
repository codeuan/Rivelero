"""Sentinel-2 data acquisition for Rivelero.

Retrieves Sentinel-2 L2A B04/B08-derived NDVI from the Copernicus Data Space
Ecosystem Process API. This module provides data; it does not interpret NDVI
as botanical suitability.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

__all__ = ["NDVI", "fetch_ndvi"]

def fetch_ndvi(
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
    mask_water: bool = True,
) -> dict[str, Any]:
    """
    Retrieve Sentinel-2 L2A NDVI from the CDSE Process API.

    Unsupported contributed adapter (see contrib/README.md): it is not wired
    into the Rivelero application or its canonical models.
    """
    _validate_fetch_parameters(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
    )

    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    client_id = client_id or os.getenv("CDSE_CLIENT_ID")
    client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Missing credentials. Pass client_id/client_secret or set "
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
        )

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    epsg = _utm_epsg_for_lonlat(center_lon, center_lat)
    bbox_projected = _project_bbox_crs84_to_epsg(
        bbox_lonlat,
        destination_epsg=epsg,
    )

    min_x, min_y, max_x, max_y = bbox_projected

    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Projected request bounds have non-positive size.")

    access_token = _get_cdse_access_token(
        client_id,
        client_secret,
        timeout=min(timeout, 30),
    )

    bad_scl_classes = [3, 7, 8, 9, 10, 11]

    if mask_water:
        bad_scl_classes.append(6)

    bad_scl_javascript = json.dumps(bad_scl_classes)

    evalscript = f"""
    //VERSION=3

    function setup() {{
        return {{
            input: ["B04", "B08", "SCL", "dataMask"],
            output: {{
                bands: 2,
                sampleType: "FLOAT32"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        let ndvi = index(sample.B08, sample.B04);
        let badSCL = {bad_scl_javascript}.includes(sample.SCL);
        let valid = (sample.dataMask === 1) && !badSCL;

        return [ndvi, valid ? 1 : 0];
    }}
    """

    request_body = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/EPSG/0/"
                        f"{epsg}"
                    )
                },
                "bbox": [min_x, min_y, max_x, max_y],
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to,
                        },
                        "maxCloudCoverage": max_cloud_coverage,
                        "mosaickingOrder": mosaicking_order,
                    },
                    "processing": {
                        "upsampling": "NEAREST",
                        "downsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "resx": pixel_size_m,
            "resy": pixel_size_m,
        },
        "evalscript": evalscript,
    }

    response = requests.post(
        CDSE_PROCESS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "image/tiff",
        },
        json=request_body,
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            "CDSE Process API request failed with status "
            f"{response.status_code}: {response.text[:2000]}"
        )

    with MemoryFile(response.content) as memory_file:
        with memory_file.open() as dataset:
            data = dataset.read()
            profile = dataset.profile.copy()
            transform = dataset.transform
            crs = dataset.crs

    if data.shape[0] < 2:
        raise RuntimeError(
            "CDSE response did not contain both NDVI and validity bands."
        )

    ndvi = data[0].astype(np.float32)
    valid_mask = data[1] > 0.5
    ndvi[~valid_mask] = np.nan

    finite = np.isfinite(ndvi)

    if finite.any():
        statistics = {
            "mean_ndvi": float(np.nanmean(ndvi)),
            "median_ndvi": float(np.nanmedian(ndvi)),
            "min_ndvi": float(np.nanmin(ndvi)),
            "max_ndvi": float(np.nanmax(ndvi)),
        }
    else:
        statistics = {
            "mean_ndvi": float("nan"),
            "median_ndvi": float("nan"),
            "min_ndvi": float("nan"),
            "max_ndvi": float("nan"),
        }

    return {
        "ndvi": ndvi,
        "ndvi_array": ndvi,
        "valid_mask": valid_mask,
        "transform": transform,
        "raster_transform": transform,
        "crs": crs,
        "raster_crs": crs,
        "bounds_projected": bbox_projected,
        "raster_bounds": bbox_projected,
        "bbox_lonlat": bbox_lonlat,
        "epsg": epsg,
        "pixel_size_m": pixel_size_m,
        "profile": profile,
        **statistics,
    }


def NDVI(
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
    mask_water: bool = True,
) -> dict[str, Any]:
    """
    Backwards-compatible alias for the original standalone NDVI entry point.
    """
    return fetch_ndvi(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        client_id=client_id,
        client_secret=client_secret,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
        timeout=timeout,
        mask_water=mask_water,
    )

def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Select a local WGS84 UTM EPSG code."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))

    return (32600 if lat >= 0 else 32700) + zone


def _project_bbox_crs84_to_epsg(
    bbox_lonlat: tuple[float, float, float, float],
    destination_epsg: int,
) -> tuple[float, float, float, float]:
    """Project a longitude/latitude bounding box into a projected CRS."""
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{destination_epsg}",
        always_xy=True,
    )

    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]

    projected = [
        transformer.transform(lon, lat)
        for lon, lat in corners
    ]

    x_values, y_values = zip(*projected, strict=True)

    return (
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
    )


def _get_cdse_access_token(
    client_id: str,
    client_secret: str,
    timeout: int,
) -> str:
    """Fetch a CDSE OAuth2 access token."""
    response = requests.post(
        CDSE_TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    access_token = payload.get("access_token")

    if not access_token:
        raise RuntimeError(
            f"No access_token in token response: {payload}"
        )

    return str(access_token)

def _validate_fetch_parameters(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    pixel_size_m: int,
    max_cloud_coverage: int,
    mosaicking_order: str,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(
            "bbox_lonlat must be "
            "(min_lon, min_lat, max_lon, max_lat) with min < max."
        )

    if not time_from.strip() or not time_to.strip():
        raise ValueError("Both time_from and time_to are required.")

    if pixel_size_m <= 0:
        raise ValueError("Pixel size must be greater than zero.")

    if not 0 <= max_cloud_coverage <= 100:
        raise ValueError(
            "Maximum cloud coverage must be between 0 and 100."
        )

    if mosaicking_order not in {"leastCC", "mostRecent"}:
        raise ValueError(
            "Mosaicking order should normally be 'leastCC' or "
            "'mostRecent'."
        )
